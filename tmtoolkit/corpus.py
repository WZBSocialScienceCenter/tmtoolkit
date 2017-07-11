# -*- coding: utf-8 -*-
import os
import codecs

from .utils import pickle_data, unpickle_file


class Corpus(object):
    def __init__(self, docs=None):
        self.docs = docs or {}

    @classmethod
    def from_files(cls, *args, **kwargs):
        return cls().add_files(*args, **kwargs)

    @classmethod
    def from_folder(cls, *args, **kwargs):
        return cls().add_folder(*args, **kwargs)

    @classmethod
    def from_pickle(cls, picklefile):
        return cls(unpickle_file(picklefile))

    def add_files(self, files, encoding='utf8', doc_label_fmt=u'{path}-{basename}', doc_label_path_join='_'):
        for fpath in files:
            text = read_lines_from_file(fpath, encoding=encoding)

            path_parts = path_recursive_split(os.path.normpath(fpath))
            if not path_parts:
                continue

            dirs, fname = path_parts[:-1], path_parts[-1]
            basename, ext = os.path.splitext(fname)
            if ext:
                ext = ext[1:]

            doclabel = doc_label_fmt.format(path=doc_label_path_join.join(dirs),
                                            basename=basename,
                                            ext=ext)

            if doclabel in self.docs:
                raise ValueError("duplicate label '%s' not allowed" % doclabel)

            self.docs[doclabel] = text

        return self

    def add_folder(self, folder, valid_extensions=('txt',), encoding='utf8', strip_folderpath_from_doc_label=True,
                   doc_label_fmt=u'{path}-{basename}', doc_label_path_join='_'):
        if type(valid_extensions) is str:
            valid_extensions = (valid_extensions,)

        for root, _, files in os.walk(folder):
            if not files:
                continue

            for fname in files:
                fpath = os.path.join(root, fname)
                text = read_lines_from_file(fpath, encoding=encoding)

                if strip_folderpath_from_doc_label:
                    dirs = path_recursive_split(root[len(folder)+1:])
                else:
                    dirs = path_recursive_split(root)
                basename, ext = os.path.splitext(fname)
                if ext:
                    ext = ext[1:]

                if valid_extensions and (not ext or ext not in valid_extensions):
                    continue

                doclabel = doc_label_fmt.format(path=doc_label_path_join.join(dirs),
                                                basename=basename,
                                                ext=ext)

                if doclabel in self.docs:
                    raise ValueError("duplicate label '%s' not allowed" % doclabel)

                self.docs[doclabel] = text

        return self

    def to_pickle(self, picklefile):
        pickle_data(self.docs, picklefile)

        return self

    def split_by_paragraphs(self, break_on_num_newlines=2, new_doc_label_fmt=u'{doc}-{parnum}'):
        tmp_docs = {}
        for dl, doc in self.docs.items():
            pars = paragraphs_from_lines(doc, break_on_num_newlines)
            for i, p in enumerate(pars):
                new_dl = new_doc_label_fmt.format(doc=dl, parnum=i+1)
                tmp_docs[new_dl] = p

        assert len(tmp_docs) >= len(self.docs)
        self.docs = tmp_docs

        return self


def read_lines_from_file(fpath, encoding):
    with codecs.open(fpath, encoding=encoding) as f:
        return f.readlines()


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


def paragraphs_from_lines(lines, break_on_num_newlines=2):
    """
    Parse list of `lines` from text file and split them into individual paragraphs. A paragraph must be divided by at
    least `break_on_num_newlines` line breaks (empty lines) from another paragraph.
    Return a list of paragraphs, each paragraph containing a string of sentences.
    """
    if type(lines) not in (tuple, list):
        raise ValueError(u"`lines` must be passed as list or tuple")

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
