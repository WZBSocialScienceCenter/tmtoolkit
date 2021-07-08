"""
Module that facilitates handling of raw text corpora.
"""

import os
import string
import codecs
import multiprocessing as mp
from random import sample
from zipfile import ZipFile
from tempfile import mkdtemp
from glob import glob
from functools import partial, wraps
from inspect import signature
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from loky import get_reusable_executor

from .preprocess._common import DEFAULT_LANGUAGE_MODELS
from .utils import pickle_data, unpickle_file, greedy_partitioning, \
    require_listlike, require_types, require_listlike_or_set, flatten_list, empty_chararray,\
    combine_sparse_matrices_columnwise, merge_dicts, merge_sets, merge_counters
from .bow.dtm import create_sparse_dtm, dtm_to_datatable, dtm_to_dataframe


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
DATAPATH = os.path.join(MODULE_PATH, 'data')


#%% Corpus class

class Corpus:
    """
    The Corpus class facilitates the handling of raw text corpora. By "raw text" we mean that the documents in the
    corpus are represented as plain text strings, i.e. they are *not* tokenized and hence not ready for token-based
    quantitative analysis. In order to tokenize and further process the raw text documents, you can pass the
    :class:`~tmtoolkit.corpus.Corpus` object to :class:`tmtoolkit.preprocess.TMPreproc` or use the functional
    preprocessing API from :mod:`tmtoolkit.preprocess`.

    This class implements :func:`dict` methods, i.e. it behaves like a Python :func:`dict` where the keys are document
    labels and values are the corresponding document texts as strings.
    """

    _BUILTIN_CORPORA_LOAD_KWARGS = {
        'en-NewsArticles': {
            'id_column': 'article_id',
            'text_column': 'text',
            'prepend_columns': ['title', 'subtitle']
        },
        'de-parlspeech-v2-sample-bundestag': {
            'id_column': 'parlspeech_row',
            'text_column': 'text',
        },
        'en-parlspeech-v2-sample-houseofcommons': {
            'id_column': 'parlspeech_row',
            'text_column': 'text',
        },
        'es-parlspeech-v2-sample-congreso': {
            'id_column': 'parlspeech_row',
            'text_column': 'text',
        },
        'nl-parlspeech-v2-sample-tweedekamer': {
            'id_column': 'parlspeech_row',
            'text_column': 'text',
        },
    }

    def __init__(self, docs=None, language=None, n_max_workers=None, workers_timeout_sec=10):
        """
        Construct a new :class:`~tmtoolkit.corpus.Corpus` object by passing a dictionary of documents with document
        label -> document text mapping. You can create an empty corpus by not passing any documents and later at them,
        e.g. with :meth:`~tmtoolkit.corpus.Corpus.add_doc`, :meth:`~tmtoolkit.corpus.Corpus.add_files` or
        :meth:`~tmtoolkit.corpus.Corpus.add_folder`.

        A Corpus object can also be created by loading data from files or folders. See the class methods
        :meth:`~tmtoolkit.corpus.Corpus.from_files()`, :meth:`~tmtoolkit.corpus.Corpus.from_folders()` and
        :meth:`~tmtoolkit.corpus.Corpus.from_pickle()`.

        :param docs: dictionary of documents with document label -> document text mapping
        """
        self.tokenized = False
        self.doc_paths = {}
        self.language = language
        self.nlp = None
        self.n_max_workers = n_max_workers or mp.cpu_count()
        self.procexec = get_reusable_executor(max_workers=self.n_max_workers, timeout=workers_timeout_sec)\
            if n_max_workers > 1 else None
        self.workers_docs = None
        self._docs = {}
        self.docs = docs or {}

        if language:
            init_corpus_language(self, language)

    def __str__(self):
        return self.__repr__()   # TODO

    def __repr__(self):
        return f'<Corpus [{self.n_docs} {"" if self.tokenized else "un"}tokenized document' \
               f'{"s" if self.n_docs > 1 else ""}]>'

    def __len__(self):
        """
        Dict method to return number of documents.

        :return: number of documents
        """
        return len(self.docs)

    def __getitem__(self, doc_label):
        """
        dict method for retrieving document with label `doc_label` via ``corpus[<doc_label>]``.
        """
        if doc_label not in self.docs:
            raise KeyError('document `%s` not found in corpus' % doc_label)
        return self.docs[doc_label]

    def __setitem__(self, doc_label, doc_text):
        """dict method for setting a document with label `doc_label` via ``corpus[<doc_label>] = <doc_text>``."""
        if not isinstance(doc_label, str) or not doc_label:
            raise KeyError('`doc_label` must be a valid non-empty string')

        if not isinstance(doc_text, str):
            raise ValueError('`doc_text` must be a string')

        self.docs[doc_label] = doc_text

    def __delitem__(self, doc_label):
        """dict method for removing a document with label `doc_label` via ``del corpus[<doc_label>]``."""
        if doc_label not in self.docs:
            raise KeyError('document `%s` not found in corpus' % doc_label)
        del self.docs[doc_label]

    def __iter__(self):
        """dict method for iterating through the document labels."""
        return self.docs.__iter__()

    def __contains__(self, doc_label):
        """dict method for checking whether `doc_label` exists in this corpus."""
        return doc_label in self.docs

    def __copy__(self):
        """
        Copy a Corpus object including all of its its present state. Performs a deep copy.
        """
        return self.copy()

    def __deepcopy__(self, memodict=None):
        """
        Copy a Corpus object including all of its its present state. Performs a deep copy.
        """
        return self.copy()

    def copy(self):
        """
        Copy a Corpus object including all of its its present state. Performs a deep copy.

        :return: copy of this Corpus object
        """
        newobj = Corpus(docs=self.docs.copy())
        newobj.doc_paths = self.doc_paths.copy()

        return newobj

    def items(self):
        """dict method to retrieve pairs of document labels and texts."""
        return self.docs.items()

    def keys(self):
        """dict method to retrieve document labels."""
        return self.docs.keys()

    def values(self):
        """dict method to retrieve document texts."""
        return self.docs.values()

    def get(self, *args):
        """dict method to retrieve a specific document like ``corpus.get(<doc_label>, <default>)``."""
        return self.docs.get(*args)

    @property
    def docs(self):
        return self._docs

    @docs.setter
    def docs(self, docs):
        self._docs = docs

        if self.n_max_workers > 1:
            docs_and_lengths = {dl: len(doc) for dl, doc in docs.items()}
            workers_doclabels = greedy_partitioning(docs_and_lengths, k=self.n_max_workers, return_only_labels=True)
            self.workers_docs = [{dl: self._docs[dl]}
                                 for doclabels in workers_doclabels
                                 for dl in doclabels]
        else:
            self.workers_docs = None

    @classmethod
    def from_files(cls, *args, **kwargs):
        """
        Construct Corpus object by loading files. See method :meth:`~tmtoolkit.corpus.Corpus.add_files()` for
        available arguments.

        :return: Corpus instance
        """
        return cls().add_files(*args, **kwargs)

    @classmethod
    def from_folder(cls, *args, **kwargs):
        """
        Construct Corpus object by loading files from a folder. See method
        :meth:`~tmtoolkit.corpus.Corpus.add_folder()` for available arguments.

        :return: Corpus instance
        """
        return cls().add_folder(*args, **kwargs)

    @classmethod
    def from_tabular(cls, *args, **kwargs):
        """
        Construct Corpus object by loading documents from a tabular file, i.e. CSV or Excel file. See method
        :meth:`~tmtoolkit.corpus.Corpus.add_tabular()` for available arguments.

        :return: Corpus instance
        """
        return cls().add_tabular(*args, **kwargs)

    @classmethod
    def from_zip(cls, *args, **kwargs):
        """
        Construct Corpus object by loading files from a ZIP file. See method
        :meth:`~tmtoolkit.corpus.Corpus.add_zip()` for available arguments.

        :return: Corpus instance
        """
        return cls().add_zip(*args, **kwargs)

    @classmethod
    def from_pickle(cls, picklefile):
        """
        Construct Corpus object by loading `picklefile`.

        :return: Corpus instance
        """
        return cls(unpickle_file(picklefile))

    @classmethod
    def from_builtin_corpus(cls, corpus_label):
        """
        Construct Corpus object by loading one of the built-in datasets specified by `corpus_label`. To get a list
        of available built-in datasets, use :meth:`~tmtoolkit.corpus.Corpus.builtin_corpora`.

        :param corpus_label: the corpus to load (one of the labels listed in
                             :meth:`~tmtoolkit.corpus.Corpus.builtin_corpora`
        :return: Corpus instance
        """

        available = cls.builtin_corpora(with_paths=True)

        if corpus_label in available:
            return cls.from_zip(available[corpus_label], **cls._BUILTIN_CORPORA_LOAD_KWARGS.get(corpus_label, {}))
        else:
            raise ValueError('built-in corpus does not exist:', corpus_label)

    @staticmethod
    def builtin_corpora(with_paths=False):
        """
        Return list of available built-in corpora.

        :param with_paths: if True, return dict mapping corpus label to absolute path to dataset, else return only
                           a list of corpus labels
        :return: dict or list, depending on `with_paths`
        """

        corpora = {}

        for fpath in glob(os.path.join(DATAPATH, '**/*.zip')):
            pathcomp = path_recursive_split(fpath)
            basename, _ = os.path.splitext(pathcomp[-1])

            corpora[pathcomp[-2] + '-' + basename] = os.path.abspath(fpath)

        if with_paths:
            return corpora
        else:
            return sorted(corpora.keys())

    @property
    def n_docs(self):
        """Number of documents."""
        return len(self)

    @property
    def doc_labels(self):
        """Sorted document labels."""
        return self.get_doc_labels(sort=True)

    @property
    def doc_lengths(self):
        """
        Return dict with number of characters per document.

        :return: dict mapping document labels to document text length in number of characters
        """
        return dict(zip(self.keys(), map(len, self.values())))

    @property
    def unique_characters(self):
        """
        Return a the set of unique characters that exist in this corpus.

        :return: set of unique characters that exist in this corpus
        """
        charset = set()
        for doc in self.docs.values():
            charset |= set(doc)

        return charset

    def get_doc_labels(self, sort=False):
        """
        Return the document labels, optionally sorted.

        :param sort: sort the document labels if True
        :return: list of document labels
        """
        labels = self.keys()

        if sort:
            return sorted(labels)
        else:
            return list(labels)

    def add_doc(self, doc_label, doc_text, force_unix_linebreaks=True):
        """
        Add a document with document label `doc_label` and text `doc_text` to the corpus.

        :param doc_label: document label string
        :param doc_text: document text string
        :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
        :return: this corpus instance
        """
        if not isinstance(doc_label, str) or not doc_label:
            raise ValueError('`doc_label` must be a valid non-empty string')

        if not isinstance(doc_text, str):
            raise ValueError('`doc_text` must be a string')

        if doc_label in self.docs:
            raise ValueError('a document with the label `%s` already exists in the corpus' % doc_label)

        if force_unix_linebreaks:
            doc_text = linebreaks_win2unix(doc_text)

        self.docs[doc_label] = doc_text

        return self

    def add_files(self, files, encoding='utf8', doc_label_fmt='{path}-{basename}', doc_label_path_join='_',
                  doc_labels=None, read_size=-1, force_unix_linebreaks=True):
        """
        Read text documents from files passed in `files` and add them to the corpus. The document label for each new
        document is determined via format string `doc_label_fmt`.

        :param files: single file string or sequence of files to read
        :param encoding: character encoding of the files
        :param doc_label_fmt: document label format string with placeholders "path", "basename", "ext"
        :param doc_label_path_join: string with which to join the components of the file paths
        :param doc_labels: instead generating document labels from `doc_label_fmt`, pass a list of document labels
                           to be used directly
        :param read_size: max. number of characters to read. -1 means read full file.
        :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
        :return: this instance
        """

        if isinstance(files, str):
            files = [files]

        require_listlike_or_set(files)

        if doc_labels is not None and len(doc_labels) != len(files) or isinstance(files, set):
            raise ValueError('`doc_labels` must be of same length as list `files`')

        for i, fpath in enumerate(files):
            text = read_text_file(fpath, encoding=encoding, read_size=read_size,
                                  force_unix_linebreaks=force_unix_linebreaks)

            path_parts = path_recursive_split(os.path.normpath(fpath))
            if not path_parts:
                continue

            dirs, fname = path_parts[:-1], path_parts[-1]
            basename, ext = os.path.splitext(fname)
            basename = basename.strip()
            if ext:
                ext = ext[1:]

            if doc_labels is None:
                doclabel = doc_label_fmt.format(path=doc_label_path_join.join(dirs),
                                                basename=basename,
                                                ext=ext)

                if doclabel.startswith('-'):
                    doclabel = doclabel[1:]
            else:
                doclabel = doc_labels[i]

            if doclabel in self.docs:
                raise ValueError("duplicate label '%s' not allowed" % doclabel)

            self.docs[doclabel] = text
            self.doc_paths[doclabel] = fpath

        return self

    def add_folder(self, folder, valid_extensions=('txt',), encoding='utf8', strip_folderpath_from_doc_label=True,
                   doc_label_fmt='{path}-{basename}', doc_label_path_join='_', read_size=-1,
                   force_unix_linebreaks=True):
        """
        Read documents residing in folder `folder` and ending on file extensions specified via `valid_extensions`.
        Note that only raw text files can be read, not PDFs, Word documents, etc. These must be converted to raw
        text files beforehand, for example with pdttotext (poppler-utils package) or pandoc.

        :param folder: Folder from where the files are read.
        :param valid_extensions: Sequence of valid file extensions like .txt, .md, etc.
        :param encoding: character encoding of the files
        :param strip_folderpath_from_doc_label: if True, do not include the folder path in the document label
        :param doc_label_fmt: document label format string with placeholders "path", "basename", "ext"
        :param doc_label_path_join: string with which to join the components of the file paths
        :param read_size: max. number of characters to read. -1 means read full file.
        :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
        :return: this instance
        """
        if not os.path.exists(folder):
            raise IOError("path does not exist: '%s'" % folder)

        if isinstance(valid_extensions, str):
            valid_extensions = (valid_extensions,)

        for root, _, files in os.walk(folder):
            if not files:
                continue

            for fname in files:
                fpath = os.path.join(root, fname)
                text = read_text_file(fpath, encoding=encoding, read_size=read_size,
                                      force_unix_linebreaks=force_unix_linebreaks)

                if strip_folderpath_from_doc_label:
                    dirs = path_recursive_split(root[len(folder)+1:])
                else:
                    dirs = path_recursive_split(root)
                basename, ext = os.path.splitext(fname)
                basename = basename.strip()
                if ext:
                    ext = ext[1:]

                if valid_extensions and (not ext or ext not in valid_extensions):
                    continue

                doclabel = doc_label_fmt.format(path=doc_label_path_join.join(dirs),
                                                basename=basename,
                                                ext=ext)
                if doclabel.startswith('-'):
                    doclabel = doclabel[1:]

                if doclabel in self.docs:
                    raise ValueError("duplicate label '%s' not allowed" % doclabel)

                self.docs[doclabel] = text
                self.doc_paths[doclabel] = fpath

        return self

    def add_tabular(self, files, id_column, text_column, prepend_columns=None, encoding='utf8',
                    doc_label_fmt='{basename}-{id}', force_unix_linebreaks=True, **kwargs):
        """
        Add documents from tabular (CSV or Excel) file(s).

        :param files: single string or list of strings with path to file(s) to load
        :param id_column: column name of document identifiers
        :param text_column: column name of document texts
        :param prepend_columns: if not None, pass a list of columns whose contents should be added before the document
                                text, e.g. ``['title', 'subtitle']``
        :param encoding: character encoding of the files
        :param doc_label_fmt: document label format string with placeholders ``"basename"``, ``"id"`` (document ID), and
                              ``"row_index"`` (dataset row index)
        :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks in texts
        :param kwargs: additional arguments passed to :func:`pandas.read_csv` or :func:`pandas.read_excel`
        :return: this instance
        """

        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError('package `pandas` must be installed to use this function')

        if isinstance(files, str):
            files = [files]

        read_opts = {
            'encoding': encoding,
            'usecols': [id_column, text_column]
        }

        if prepend_columns:
            require_listlike(prepend_columns)
            read_opts['usecols'] += prepend_columns

        if all(isinstance(x, int) for x in read_opts['usecols']):
            id_column, text_column = 0, 1
            if prepend_columns:
                prepend_columns = list(range(2, len(prepend_columns) + 2))

        read_opts.update(kwargs)

        read_opts_excel = read_opts.copy()
        del read_opts_excel['encoding']

        for fpath in files:
            if fpath.endswith('.csv'):
                data = pd.read_csv(fpath, **read_opts)
            elif fpath.endswith('.xls') or fpath.endswith('.xlsx'):
                if fpath.endswith('.xlsx') and 'engine' not in read_opts_excel:
                    read_opts_excel['engine'] = 'openpyxl'
                data = pd.read_excel(fpath, **read_opts_excel)
            else:
                raise ValueError('only file extensions ".csv", ".xls" and ".xlsx" are supported')

            basename, _ = os.path.splitext(fpath)
            basename = os.path.basename(basename).strip()

            for idx, row in data.iterrows():
                doclabel = doc_label_fmt.format(basename=basename, id=row[id_column], row_index=idx)

                if doclabel in self.docs:
                    raise ValueError("duplicate label '%s' not allowed" % doclabel)

                if prepend_columns:
                    text = '\n\n'.join([row[col] for col in (prepend_columns + [text_column]) if pd.notna(row[col])])
                else:
                    text = row[text_column] if pd.notna(row[text_column]) else ''

                if force_unix_linebreaks:
                    text = linebreaks_win2unix(text)

                self.docs[doclabel] = text
                self.doc_paths[doclabel] = fpath + ':' + str(idx)

        return self

    def add_zip(self, zipfile, valid_extensions=('txt', 'csv', 'xls', 'xlsx'), encoding='utf8',
                doc_label_fmt_txt='{path}-{basename}', doc_label_path_join='_', doc_label_fmt_tabular='{basename}-{id}',
                force_unix_linebreaks=True, **kwargs):
        """
        Add documents from a ZIP file. The ZIP file may include documents with extensions listed in `valid_extensions`.

        For file extensions 'csv', 'xls' or 'xlsx' :meth:`~tmtoolkit.corpus.Corpus.add_tabular()` will be called. Make
        sure to pass at least the parameters `id_column` and `text_column` as additional `kwargs` if your ZIP contains
        such files.

        For all other file extensions :meth:`~tmtoolkit.corpus.Corpus.add_files()` will be called.

        :param zipfile: path to ZIP file to be loaded; string
        :param valid_extensions: list of valid file extensions of ZIP file members; all other members will be ignored
        :param encoding: character encoding of the files
        :param doc_label_fmt_txt: document label format for non-tabular files; string with placeholders ``"path"``,
                                  ``"basename"``, ``"ext"``
        :param doc_label_path_join: string with which to join the components of the file paths
        :param doc_label_fmt_tabular: document label format string for tabular files; placeholders ``"basename"``,
                                      ``"id"`` (document ID), and ``"row_index"`` (dataset row index)
        :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks in texts
        :param kwargs: additional arguments passed to :meth:`~tmtoolkit.corpus.Corpus.add_tabular()` or
                       :meth:`~tmtoolkit.corpus.Corpus.add_files()`
        :return: this instance
        """

        tmpdir = mkdtemp()

        read_size = kwargs.pop('read_size', -1)

        with ZipFile(zipfile) as zipobj:
            for member in zipobj.namelist():
                path_parts = path_recursive_split(member)

                if not path_parts:
                    continue

                dirs, fname = path_parts[:-1], path_parts[-1]

                basename, ext = os.path.splitext(fname)
                basename = basename.strip()

                if ext:
                    ext = ext[1:]

                if ext in valid_extensions:
                    tmpfile = zipobj.extract(member, tmpdir)

                    if ext in {'csv', 'xls', 'xlsx'}:
                        self.add_tabular(tmpfile, encoding=encoding, doc_label_fmt=doc_label_fmt_tabular,
                                         force_unix_linebreaks=force_unix_linebreaks, **kwargs)
                    else:
                        doclabel = doc_label_fmt_txt.format(path=doc_label_path_join.join(dirs),
                                                            basename=basename,
                                                            ext=ext)

                        if doclabel.startswith('-'):
                            doclabel = doclabel[1:]

                        self.add_files(tmpfile, doc_labels=[doclabel], encoding=encoding, read_size=read_size,
                                       force_unix_linebreaks=force_unix_linebreaks)

        return self

    def to_pickle(self, picklefile):
        """
        Save corpus to pickle file `picklefile`.

        :param picklefile: path to file to store corpus
        :return: this instance
        """

        pickle_data(self.docs, picklefile)

        return self

    def split_by_paragraphs(self, break_on_num_newlines=2, splitchar='\n', join_paragraphs=1,
                            force_unix_linebreaks=True, new_doc_label_fmt='{doc}-{parnum}'):
        """
        Split documents in corpus by paragraphs and set the resulting documents as new corpus.

        :param break_on_num_newlines: Threshold of minimum number of linebreaks that denote a new paragraph.
        :param splitchar: Linebreak character(s)
        :param join_paragraphs: Number of subsequent paragraphs to join and form a document
        :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
        :param new_doc_label_fmt: document label format string with placeholders "doc" and "parnum" (paragraph number)
        :return: this corpus instance
        """

        if join_paragraphs < 1:
            raise ValueError('`join_paragraphs` must be at least 1')

        tmp_docs = {}

        if join_paragraphs > 1:
            glue = '\n' * break_on_num_newlines
        else:
            glue = ''

        tmp_doc_paths = {}
        for dl, doc in self.docs.items():
            doc_path = self.doc_paths.get(dl, None)
            pars = paragraphs_from_lines(doc, splitchar=splitchar, break_on_num_newlines=break_on_num_newlines,
                                         force_unix_linebreaks=force_unix_linebreaks)
            i = 1
            cur_ps = []
            for parnum, p in enumerate(pars):
                cur_ps.append(p)
                if i == join_paragraphs:
                    p_joined = glue.join(cur_ps)
                    new_dl = new_doc_label_fmt.format(doc=dl, parnum=parnum+1)
                    tmp_docs[new_dl] = p_joined

                    if doc_path:
                        tmp_doc_paths[new_dl] = doc_path

                    i = 1
                    cur_ps = []
                else:
                    i += 1
        assert len(tmp_docs) >= len(self.docs)
        self.docs = tmp_docs
        self.doc_paths = tmp_doc_paths

        return self

    def sample(self, n, inplace=False, as_corpus=True):
        """
        Return a sample of `n` documents` of this corpus. Sampling occurs without replacement.

        :param n: sample size
        :param inplace: replace this corpus' documents with the sampled documents if this argument is True
        :param as_corpus: if True, return result as new `Corpus` object, else as dict. Only applies when `inplace`
                          is False
        :return: a sample of `n` documents` as dict if `inplace` is False, else this instance with resampled documents
        """
        if not self.docs:
            return ValueError('cannot sample from empty corpus')

        if not 1 <= n <= len(self.docs):
            return ValueError('`n` must be between 1 and %d' % len(self.docs))

        sampled_docs = {dl: self.docs[dl] for dl in sample(self.docs.keys(), n)}

        if inplace:
            self.docs = sampled_docs
            return self
        else:
            if as_corpus:
                return Corpus(sampled_docs)
            else:
                return sampled_docs

    def filter_by_min_length(self, nchars):
        """
        Filter corpus by retaining only documents with at least `nchars` characters.

        :param nchars: minimum number of characters
        :return: this instance
        """
        
        self.docs = self._filter_by_length(nchars, 'min')
        return self

    def filter_by_max_length(self, nchars):
        """
        Filter corpus by retaining only documents with at most `nchars` characters.

        :param nchars: maximum number of characters
        :return: this instance
        """

        self.docs = self._filter_by_length(nchars, 'max')
        return self

    def filter_characters(self, allow_chars=string.printable, drop_chars=None):
        """
        Filter the document strings by removing all characters but those in `allow_chars` or, if `allow_chars` evaluates
        to False, remove those in `drop_chars`.

        :param allow_chars: set (like ``{'a', 'b', 'c'}`` or string sequence (like ``'abc'``)
        :param drop_chars: set or string sequence of characters to remove (if `allow_chars` evaluates to False)
        :return: this instance
        """

        if allow_chars is not None:
            if not isinstance(allow_chars, set):
                allow_chars = set(allow_chars)

            drop_chars = ''.join(self.unique_characters - allow_chars)
        else:
            if isinstance(drop_chars, (set, list, tuple)):
                drop_chars = ''.join(drop_chars)

            if not isinstance(drop_chars, str):
                raise ValueError('`drop_chars` must be a sequence, set or string if `allow_chars` is not given')

        return self.replace_characters(str.maketrans(drop_chars, drop_chars, drop_chars))

    def remove_characters(self, drop_chars):
        """
        Shortcut for :meth:`~tmtoolkit.corpus.Corpus.filter_characters` for removing characters in `drop_chars`.

        :param drop_chars: set or string sequence of characters to remove
        :return: this instance
        """
        return self.filter_characters(allow_chars=None, drop_chars=drop_chars)

    def replace_characters(self, translation_table):
        """
        Replace all characters in all document strings by applying the translation table `translation_table`, which
        in effect converts or removes characters.

        :param translation_table: a `dict` with character -> replacement mapping; if "replacement" None, remove that
                                  character; both "character" and "replacement" can be either single characters or
                                  ordinals; can be constructed with :func:`str.maketrans()`;
                                  Examples: ``{'a': 'X', 'b': None}`` (turns all a's to X's and removes all b's), which
                                  is equivalent to ``{97: 88, 98: None}``
        :return: this instance
        """
        def char2ord(c):
            return ord(c) if isinstance(c, str) else c

        translation_table = {char2ord(c): char2ord(r) for c, r in translation_table.items()}

        new_docs = {}
        for dl, dt in self.docs.items():
            new_docs[dl] = dt.translate(translation_table)
        self.docs = new_docs

        return self

    def apply(self, func):
        """
        Apply function `func` to each document in the corpus.

        :param func: function accepting a document text string as only argument
        :return: this instance
        """
        if not callable(func):
            raise ValueError('`func` must be callable')

        new_docs = {}
        for dl, dt in self.docs.items():
            new_docs[dl] = func(dt)
        self.docs = new_docs

        return self

    def _filter_by_length(self, nchars, predicate):
        """
        Helper function to filter corpus by minimum or maximum number of characters `nchars`.

        :param nchars: minimum or maximum number of characters `nchars`
        :param predicate: "min" or "max"
        :return: dict of filtered documents
        """
        if nchars < 0:
            raise ValueError("`nchars` must be positive")
        assert predicate in ('min', 'max')

        doc_lengths = self.doc_lengths

        filtered_docs = {}
        for dl, dt in self.docs.items():
            len_doc = doc_lengths[dl]
            if (predicate == 'min' and len_doc >= nchars) or (predicate == 'max' and len_doc <= nchars):
                filtered_docs[dl] = dt

        return filtered_docs


#%% Corpus helper functions


def read_text_file(fpath, encoding, read_size=-1, force_unix_linebreaks=True):
    """
    Read the text file at path `fpath` with character encoding `encoding` and return it as string.

    :param fpath: path to file to read
    :param encoding: character encoding
    :param read_size: max. number of characters to read. -1 means read full file.
    :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
    :return: file content as string
    """
    with codecs.open(fpath, encoding=encoding) as f:
        contents = f.read(read_size)

        if read_size > 0:
            contents = contents[:read_size]

        if force_unix_linebreaks:
            contents = linebreaks_win2unix(contents)

        return contents


def path_recursive_split(path, base=None):
    """
    Split path `path` into its components::

        path_recursive_split('a/simple/test.txt')
        # ['a', 'simple', 'test.txt']

    :param path: a file path
    :param base: path remainder (used for recursion)
    :return: components of the path as list
    """
    if not base:
        base = []

    if os.path.isabs(path):
        path = path[1:]

    start, end = os.path.split(path)

    if end:
        base.insert(0, end)

    if start:
        return path_recursive_split(start, base=base)
    else:
        return base


def paragraphs_from_lines(lines, splitchar='\n', break_on_num_newlines=2, force_unix_linebreaks=True):
    """
    Take string of `lines`, split into list of lines using `splitchar` (or don't if `splitchar` evaluates to False) and
    then split them into individual paragraphs. A paragraph must be divided by at
    least `break_on_num_newlines` line breaks (empty lines) from another paragraph.
    Return a list of paragraphs, each paragraph containing a string of sentences.

    :param lines: either a string which will be split into lines by `splitchar` or a list of strings representing lines;
                  in this case, set `splitchar` to None
    :param splitchar: character used to split string `lines` into separate lines
    :param break_on_num_newlines: threshold of consecutive line breaks for creating a new paragraph
    :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
    :return: list of paragraphs, each paragraph containing a string of sentences
    """
    if splitchar:
        if force_unix_linebreaks:
            lines = linebreaks_win2unix(lines)

        lines = lines.split(splitchar)
    else:
        if type(lines) not in (tuple, list):
            raise ValueError("`lines` must be passed as list or tuple if `splitchar` evaluates to False")

    n_lines = len(lines)
    paragraphs = []
    n_emptylines = 0
    cur_par = ''
    # iterate through all lines
    for i, l in enumerate(lines):
        if not splitchar and force_unix_linebreaks:
            l = linebreaks_win2unix(l)

        if l.strip():
            if not cur_par:
                cur_par = l
            else:
                cur_par += ' ' + l
            n_emptylines = 0
        else:
            n_emptylines += 1

        if (n_emptylines >= break_on_num_newlines-1 or i == n_lines-1) and cur_par:
            paragraphs.append(cur_par)
            cur_par = ''
            n_emptylines = 0

    return paragraphs


def linebreaks_win2unix(text):
    """
    Convert Windows line breaks ``'\\r\\n'`` to Unix line breaks ``'\\n'``.

    :param text: text string
    :return: text string with Unix line breaks
    """
    return text.replace('\r\n', '\n')


#%% decorators


def parallelexec(collect_fn):
    def deco_fn(fn):
        @wraps(fn)
        def inner_fn(docs, *args, **kwargs):
            if isinstance(docs, Corpus) and docs.procexec:
                print(f'{os.getpid()}: distributing function {fn} for {len(docs)} docs')
                if args:
                    fn_argnames = list(signature(fn).parameters.keys())
                    # first argument in `fn` is always the documents dict -> we skip this
                    if len(fn_argnames) <= len(args):
                        raise ValueError(f'function {fn} does not accept enough additional arguments')
                    kwargs.update({fn_argnames[i+1]: v for i, v in enumerate(args)})
                res = docs.procexec.map(partial(fn, **kwargs), docs.workers_docs)
                return collect_fn(res)
            else:
                print(f'{os.getpid()}: directly applying function {fn} to {len(docs)} docs')
                return fn(docs, *args, **kwargs)

        return inner_fn

    return deco_fn


def require_tokenized(fn):
    @wraps(fn)
    def inner_fn(docs, *args, **kwargs):
        if isinstance(docs, Corpus):
            if not docs.tokenized:
                raise ValueError('argument `docs` must be tokenized')
        elif isinstance(docs, dict):
            if docs:
                first_doc = next(iter(docs.values()))
                if not isinstance(first_doc, Doc):
                    raise ValueError('objects in `docs` must be SpaCy Doc objects')
        else:
            raise ValueError('argument `docs` must be dict or Corpus')

        return fn(docs, *args, **kwargs)

    return inner_fn


#%% initialization and tokenization


def init_corpus_language(corpus, language=None, language_model=None, **spacy_opts):
    if language is None and language_model is None:
        raise ValueError('either `language` or `language_model` must be given')

    if language_model is None:
        if not isinstance(language, str) or len(language) != 2:
            raise ValueError('`language` must be a two-letter ISO 639-1 language code')

        if language not in DEFAULT_LANGUAGE_MODELS:
            raise ValueError('language "%s" is not supported' % language)
        language_model = DEFAULT_LANGUAGE_MODELS[language] + '_sm'

    spacy_kwargs = dict(disable=['parser', 'ner'])
    spacy_kwargs.update(spacy_opts)

    corpus.nlp = spacy.load(language_model, **spacy_kwargs)
    corpus.language = language


def tokenize(corpus):
    if not isinstance(corpus, Corpus):
        raise ValueError('`corpus` must be a Corpus object')
    if not corpus.nlp:
        raise ValueError('`corpus` must have an initialized SpaCy NLP pipeline')

    tokenizerpipe = corpus.nlp.pipe(corpus.values(), n_process=corpus.n_max_workers)
    corpus.docs = {dl: _init_doc(dt, doc_label=dl) for dl, dt in zip(corpus.keys(), tokenizerpipe)}
    corpus.tokenized = True

    return corpus


# def tokendocs2spacydocs(docs, vocab=None, doc_labels=None, return_vocab=False):
#     """
#     Create new spaCy documents from token lists in `docs`.
#
#     .. note:: spaCy doesn't handle empty tokens (`""`), hence these tokens will not appear in the resulting spaCy
#               documents if they exist in the input documents.
#
#     :param docs: list of document tokens
#     :param vocab: provide vocabulary to be used when generating spaCy documents; if no vocabulary is given, it will be
#                   generated from `docs`
#     :param doc_labels: optional list of document labels; if given, must be of same length as `docs`
#     :param return_vocab: if True, additionally return generated vocabulary as spaCy `Vocab` object
#     :return: list of spaCy documents or tuple with additional generated vocabulary if `return_vocab` is True
#     """
#     require_tokendocs(docs)
#
#     if doc_labels is not None and len(doc_labels) != len(docs):
#         raise ValueError('`doc_labels` must have the same length as `docs`')
#
#     if vocab is None:
#         vocab = Vocab(strings=list(vocabulary(docs) - {''}))
#     else:
#         vocab = Vocab(strings=vocab.tolist() if isinstance(vocab, np.ndarray) else list(vocab))
#
#     spacydocs = []
#     for i, tokdoc in enumerate(docs):
#         spacydocs.append(spacydoc_from_tokens(tokdoc, vocab=vocab, label=None if doc_labels is None else doc_labels[i]))
#
#     if return_vocab:
#         return spacydocs, vocab
#     else:
#         return spacydocs


#%% functions that operate on Corpus objects


@require_tokenized
@parallelexec(collect_fn=merge_dicts)
def doc_tokens(docs, to_lists=False):
    """
    If `docs` is a list of spaCy documents, return the (potentially filtered) tokens from these documents as list of
    string tokens, otherwise return the input list as-is.

    :param docs: list of string tokens or spaCy documents
    :param to_lists: if `docs` is list of spaCy documents or list of NumPy arrays, convert result to lists
    :return: list of string tokens as NumPy arrays (default) or lists (if `to_lists` is True)
    """
    if to_lists:
        fn = partial(_filtered_doc_tokens, as_list=True)
    else:
        fn = _filtered_doc_tokens
    return {dl: fn(dt) for dl, dt in docs.items()}


@require_tokenized
def vocabulary(docs, sort=False):
    """
    Return vocabulary, i.e. set of all tokens that occur at least once in at least one of the documents in `docs`.

    :param docs: list of string tokens or spaCy documents
    :param sort: return as sorted list
    :return: either set of token strings or sorted list if `sort` is True
    """
    v = set(flatten_list(doc_tokens(docs, to_lists=True).values()))

    if sort:
        if v:
            return np.array(sorted(v))
        else:
            return empty_chararray()
    else:
        return v


@require_tokenized
@parallelexec(collect_fn=merge_dicts)
def doc_lengths(docs):
    """
    Return document length (number of tokens in doc.) for each document.

    :param docs: list of string tokens or spaCy documents
    :return: list of document lengths
    """
    return {dl: len(dt) for dl, dt in doc_tokens(docs).items()}


@require_tokenized
@parallelexec(collect_fn=merge_counters)
def vocabulary_counts(docs):
    """
    Return :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
    all documents.

    :param docs: list of string tokens or spaCy documents
    :return: :class:`collections.Counter()` instance of vocabulary containing counts of occurrences of tokens across
             all documents
    """
    return Counter(flatten_list(doc_tokens(docs).values()))


@parallelexec(collect_fn=merge_counters)
def _doc_frequencies(docs):
    doc_freqs = Counter()

    for dtok in docs.values():
        for t in set(_filtered_doc_tokens(dtok, as_list=True)):
            doc_freqs[t] += 1

    return doc_freqs


@require_tokenized
def doc_frequencies(docs, proportions=False):
    """
    Document frequency per vocabulary token as dict with token to document frequency mapping.
    Document frequency is the measure of how often a token occurs *at least once* in a document.
    Example with absolute document frequencies:

    .. code-block:: text

        doc tokens
        --- ------
        A   z, z, w, x
        B   y, z, y
        C   z, z, y, z

        document frequency df(z) = 3  (occurs in all 3 documents)
        df(x) = df(w) = 1 (occurs only in A)
        df(y) = 2 (occurs in B and C)
        ...

    :param docs: list of string tokens or spaCy documents
    :param proportions: if True, normalize by number of documents to obtain proportions
    :return: dict mapping token to document frequency
    """
    doc_freqs = _doc_frequencies(docs)

    if proportions:
        n_docs = len(docs)
        return {w: n/n_docs for w, n in doc_freqs.items()}
    else:
        return doc_freqs


@require_tokenized
@parallelexec(collect_fn=merge_dicts)
def ngrams(docs, n, join=True, join_str=' '):
    """
    Generate and return n-grams of length `n`.

    :param docs: list of string tokens or spaCy documents
    :param n: length of n-grams, must be >= 2
    :param join: if True, join generated n-grams by string `join_str`
    :param join_str: string used for joining
    :return: list of n-grams; if `join` is True, the list contains strings of joined n-grams, otherwise the list
             contains lists of size `n` in turn containing the strings that make up the n-gram
    """
    return {dl: _ngrams_from_tokens(dt, n=n, join=join, join_str=join_str)
            for dl, dt in doc_tokens(docs, to_lists=True).items()}


@require_tokenized
def dtm(docs, as_datatable=False, as_dataframe=False, dtype=None):
    if len(docs) > 0:
        w_dtms, w_doc_labels, w_vocab = zip(*_sparse_dtms(docs))
        dtm, vocab, dtm_doc_labels = combine_sparse_matrices_columnwise(w_dtms, w_vocab, w_doc_labels,
                                                                            dtype=dtype)
        # sort according to document labels
        dtm = dtm[np.argsort(dtm_doc_labels), :]
        doc_labels = np.sort(dtm_doc_labels)
    else:
        dtm = csr_matrix((0, 0), dtype=dtype or int)   # empty sparse matrix
        vocab = empty_chararray()
        doc_labels = empty_chararray()

    if as_datatable:
        return dtm_to_datatable(dtm, doc_labels, vocab)
    elif as_dataframe:
        return dtm_to_dataframe(dtm, doc_labels, vocab)
    else:
        return dtm


#%% functions that operate on a single spacy document

def spacydoc_from_tokens(tokens, vocab=None, spaces=None, lemmata=None, label=None):
    """
    Create a new spaCy ``Doc`` document with tokens `tokens`.

    :param tokens: list, tuple or NumPy array of string tokens
    :param vocab: list, tuple, set, NumPy array or spaCy ``Vocab`` object with vocabulary; if None, vocabulary will be
                  generated from `tokens`
    :param spaces: list, tuple or NumPy array of whitespace for each token
    :param lemmata: list, tuple or NumPy array of string lemmata for each token
    :param label: document label
    :return: spaCy ``Doc`` document
    """
    require_types(tokens, (tuple, list, np.ndarray), error_msg='the argument must be a list, tuple or NumPy array')

    tokens = [t for t in (tokens.tolist() if isinstance(tokens, np.ndarray) else tokens)
              if t]    # spaCy doesn't accept empty tokens and also no NumPy "np.str_" type tokens

    if vocab is None:
        vocab = Vocab(strings=list(set(tokens)))
    elif not isinstance(vocab, Vocab):
        vocab = Vocab(strings=vocab.tolist() if isinstance(vocab, np.ndarray) else list(vocab))

    if lemmata is not None and len(lemmata) != len(tokens):
        raise ValueError('`lemmata` must have the same length as `tokens`')

    new_doc = Doc(vocab, words=tokens, spaces=spaces)
    assert len(new_doc) == len(tokens)

    _init_doc(new_doc, doc_label=label)

    if lemmata is not None:
        lemmata = lemmata.tolist() if isinstance(lemmata, np.ndarray) else lemmata
        for t, lem in zip(new_doc, lemmata):
            t.lemma_ = lem

    return new_doc


#%% helper functions


def _filtered_doc_tokens(doc, as_list=False):
    if len(doc) == 0 or doc.user_data['mask'].sum() == 0:
        return [] if as_list else empty_chararray()

    res = np.asarray([t.text for t in doc])[doc.user_data['mask']]
    return res.tolist() if as_list else res


def _init_doc(doc, mask=None, doc_label=None):   # tokens=None
    assert isinstance(doc, Doc)

    if doc_label is not None:
        doc._.label = doc_label

    if mask is not None:
        assert isinstance(mask, np.ndarray)
        assert mask.dtype.kind == 'b'
        assert len(doc) == len(mask)

    doc.user_data['mask'] = mask if isinstance(mask, np.ndarray) else np.repeat(True, len(doc))

    return doc


def _ngrams_from_tokens(tokens, n, join=True, join_str=' '):
    """
    Helper function to produce ngrams of length `n` from a list of string tokens `tokens`.

    :param tokens: list of string tokens
    :param n: size of ngrams
    :param join: if True, join each ngram by `join_str`, i.e. return list of ngram strings; otherwise return list of
                 ngram lists
    :param join_str: if `join` is True, use this string to join the parts of the ngrams
    :return: return list of ngram strings if `join` is True, otherwise list of ngram lists
    """
    if n < 2:
        raise ValueError('`n` must be at least 2')

    if len(tokens) == 0:
        return []

    if len(tokens) < n:
        # raise ValueError('`len(tokens)` should not be smaller than `n`')
        ngrams = [tokens]
    else:
        ngrams = [[tokens[i+j] for j in range(n)]
                  for i in range(len(tokens)-n+1)]
    if join:
        return list(map(lambda x: join_str.join(x), ngrams))
    else:
        return ngrams


@parallelexec(collect_fn=list)
def _sparse_dtms(docs):
    print(f'DOCS: {docs}')
    vocab = vocabulary(docs, sort=True)
    tokens = list(doc_tokens(docs).values())
    alloc_size = sum(len(set(dtok)) for dtok in tokens)  # sum of *unique* tokens in each document

    print(f'VOCAB: {vocab}')
    print(f'TOKENS: {tokens}')

    return (create_sparse_dtm(vocab, tokens, alloc_size, vocab_is_sorted=True),
            docs.keys(),
            vocab)
