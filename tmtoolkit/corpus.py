"""
Module that facilitates handling of raw text corpora.
"""

import os
import string
import codecs
from random import sample

from .utils import pickle_data, unpickle_file, require_listlike_or_set

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
    def __init__(self, docs=None):
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
        self.docs = docs or {}
        self.doc_paths = {}

    def __str__(self):
        return 'Corpus with %d documents' % self.n_docs

    def __repr__(self):
        return '<Corpus [%d documents]>' % self.n_docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, doc_label):
        """dict method for retrieving document with label `doc_label` via ``corpus[<doc_label>]``."""
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

    @classmethod
    def from_files(cls, *args, **kwargs):
        """
        Construct Corpus object by loading files. See method :meth:`~tmtoolkit.corpus.Corpus.add_files()` for
        available arguments.
        """
        return cls().add_files(*args, **kwargs)

    @classmethod
    def from_folder(cls, *args, **kwargs):
        """
        Construct Corpus object by loading files from a folder. See method
        :meth:`~tmtoolkit.corpus.Corpus.add_folder()` for available arguments.
        """
        return cls().add_folder(*args, **kwargs)

    @classmethod
    def from_pickle(cls, picklefile):
        """Construct Corpus object by loading `picklefile`."""
        return cls(unpickle_file(picklefile))

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
        """dict with document labels -> document text length in number of characters."""
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

    def get_doc_labels(self, sort=True):
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
                  read_size=-1, force_unix_linebreaks=True):
        """
        Read text documents from files passed in `files` and add them to the corpus. The document label for each new
        document is determined via format string `doc_label_fmt`.

        :param files: sequence of files to read
        :param encoding: character encoding of the files
        :param doc_label_fmt: document label format string with placeholders "path", "basename", "ext"
        :param doc_label_path_join: string with which to join the components of the file paths
        :param read_size: max. number of characters to read. -1 means read full file.
        :param force_unix_linebreaks: if True, convert Windows linebreaks to Unix linebreaks
        :return: this instance
        """
        require_listlike_or_set(files)

        for fpath in files:
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

    def sample(self, n, inplace=False):
        """
        Return a sample of `n` documents` of this corpus. Sampling occurs without replacement.

        :param n: sample size
        :param inplace: Replace this corpus' documents with the sampled documents if this argument is True
        :return:  a sample of `n` documents` as dict if `inplace` is False, else this instance with resampled documents
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

    def filter_characters(self, allow_chars=string.printable):
        """
        Filter the document strings by removing all characters but those in `allow_chars`.

        :param allow_chars: set (like ``{'a', 'b', 'c'}`` or string sequence (like ``'abc'``)
        :return: this instance
        """
        if not isinstance(allow_chars, set):
            allow_chars = set(allow_chars)

        drop_chars = ''.join(self.unique_characters - allow_chars)

        return self.replace_characters(str.maketrans(drop_chars, drop_chars, drop_chars))

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


#%% Helper functions


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
