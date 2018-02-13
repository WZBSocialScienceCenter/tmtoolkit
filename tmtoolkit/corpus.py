# -*- coding: utf-8 -*-
import os
import codecs
from random import sample

import six

from .utils import pickle_data, unpickle_file, require_listlike


class Corpus(object):
    def __init__(self, docs=None):
        self.docs = docs or {}
        self.doc_paths = {}

    def __str__(self):
        return 'Corpus with %d documents' % len(self)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, doc_label):
        if doc_label not in self.docs:
            raise KeyError('document `%s` not found in corpus' % doc_label)
        return self.docs[doc_label]

    def __setitem__(self, doc_label, doc_text):
        if not isinstance(doc_label, six.string_types) or not doc_label:
            raise KeyError('`doc_label` must be a valid non-empty string')

        if not isinstance(doc_text, six.string_types):
            raise ValueError('`doc_text` must be a string')

        self.docs[doc_label] = doc_text

    def __delitem__(self, doc_label):
        if doc_label not in self.docs:
            raise KeyError('document `%s` not found in corpus' % doc_label)
        del self.docs[doc_label]

    def __iter__(self):
        return self.docs.__iter__()

    def __contains__(self, doc_label):
        return doc_label in self.docs

    def items(self):
        return self.docs.items()

    def keys(self):
        return self.docs.keys()

    def get(self, *args):
        return self.docs.get(*args)

    @classmethod
    def from_files(cls, *args, **kwargs):
        return cls().add_files(*args, **kwargs)

    @classmethod
    def from_folder(cls, *args, **kwargs):
        return cls().add_folder(*args, **kwargs)

    @classmethod
    def from_pickle(cls, picklefile):
        return cls(unpickle_file(picklefile))

    def get_doc_labels(self, sort=True):
        labels = self.keys()

        if sort:
            return sorted(labels)
        else:
            return labels

    def add_doc(self, doc_label, doc_text):
        if not isinstance(doc_label, six.string_types) or not doc_label:
            raise ValueError('`doc_label` must be a valid non-empty string')

        if not isinstance(doc_text, six.string_types):
            raise ValueError('`doc_text` must be a string')

        if doc_label in self.docs:
            raise ValueError('a document with the label `%s` already exists in the corpus' % doc_label)

        self.docs[doc_label] = doc_text

    def add_files(self, files, encoding='utf8', doc_label_fmt=u'{path}-{basename}', doc_label_path_join='_',
                  read_size=-1):
        require_listlike(files)

        for fpath in files:
            text = read_full_file(fpath, encoding=encoding, read_size=read_size)

            path_parts = path_recursive_split(os.path.normpath(fpath))
            if not path_parts:
                continue

            dirs, fname = path_parts[:-1], path_parts[-1]
            basename, ext = os.path.splitext(fname)
            basename = basename.strip()
            if ext:
                ext = ext[1:]

            doclabel_path = six.u(doc_label_path_join.join(dirs))
            doclabel_basename = six.u(basename)
            doclabel = doc_label_fmt.format(path=doclabel_path,
                                            basename=doclabel_basename,
                                            ext=ext)

            if doclabel.startswith('-'):
                doclabel = doclabel[1:]

            if doclabel in self.docs:
                raise ValueError("duplicate label '%s' not allowed" % doclabel)

            self.docs[doclabel] = text
            self.doc_paths[doclabel] = fpath

        return self

    def add_folder(self, folder, valid_extensions=('txt',), encoding='utf8', strip_folderpath_from_doc_label=True,
                   doc_label_fmt=u'{path}-{basename}', doc_label_path_join='_', read_size=-1):
        if not os.path.exists(folder):
            raise IOError("path does not exist: '%s'" % folder)

        if isinstance(valid_extensions, six.string_types):
            valid_extensions = (valid_extensions,)

        for root, _, files in os.walk(folder):
            if not files:
                continue

            for fname in files:
                fpath = os.path.join(root, fname)
                text = read_full_file(fpath, encoding=encoding, read_size=read_size)

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

                doclabel_path = six.u(doc_label_path_join.join(dirs))
                doclabel_basename = six.u(basename)
                doclabel = doc_label_fmt.format(path=doclabel_path,
                                                basename=doclabel_basename,
                                                ext=ext)
                if doclabel.startswith('-'):
                    doclabel = doclabel[1:]

                if doclabel in self.docs:
                    raise ValueError("duplicate label '%s' not allowed" % doclabel)

                self.docs[doclabel] = text
                self.doc_paths[doclabel] = fpath

        return self

    def to_pickle(self, picklefile):
        pickle_data(self.docs, picklefile)

        return self

    def split_by_paragraphs(self, break_on_num_newlines=2, join_paragraphs=1, new_doc_label_fmt=u'{doc}-{parnum}'):
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
            pars = paragraphs_from_lines(doc, break_on_num_newlines=break_on_num_newlines)
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

    def sample(self, n):
        if not self.docs:
            return ValueError('cannot sample from empty corpus')

        if not 1 <= n <= len(self.docs):
            return ValueError('`n` must be between 1 and %d' % len(self.docs))

        tmp = {dl: self.docs[dl] for dl in sample(self.docs.keys(), n)}
        self.docs = tmp

        return self

    def filter_by_min_length(self, nchars):
        self.docs = self._filter_by_length(nchars, 'min')
        return self

    def filter_by_max_length(self, nchars):
        self.docs = self._filter_by_length(nchars, 'max')
        return self

    def _filter_by_length(self, nchars, predicate):
        if nchars < 0:
            raise ValueError("`nchars` must be positive")
        assert predicate in ('min', 'max')

        filtered_docs = {}
        for dl, dt in self.docs.items():
            if (predicate == 'min' and len(dt) >= nchars) or (predicate == 'max' and len(dt) <= nchars):
                filtered_docs[dl] = dt

        return filtered_docs


def read_full_file(fpath, encoding, read_size=-1):
    with codecs.open(fpath, encoding=encoding) as f:
        contents = f.read(read_size)
        if read_size > 0:
            return contents[:read_size]
        else:
            return contents


def path_recursive_split(path, base=None):
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


def paragraphs_from_lines(lines, splitchar='\n', break_on_num_newlines=2):
    """
    Take string of `lines`, split into list of lines using `splitchar` (or don't if `splitchar` evaluates to False) and
    then split them into individual paragraphs. A paragraph must be divided by at
    least `break_on_num_newlines` line breaks (empty lines) from another paragraph.
    Return a list of paragraphs, each paragraph containing a string of sentences.
    """
    if splitchar:
        lines = lines.split(splitchar)
    else:
        if type(lines) not in (tuple, list):
            raise ValueError(u"`lines` must be passed as list or tuple if `splitchar` evaluates to False")

    n_lines = len(lines)
    paragraphs = []
    n_emptylines = 0
    cur_par = ''
    # iterate through all lines
    for i, l in enumerate(lines):
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
