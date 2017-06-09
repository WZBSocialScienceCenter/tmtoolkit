# -*- coding: utf-8 -*-
import os

from .utils import pickle_data, unpickle_file


class Corpus(object):
    def __init__(self, docs=None):
        self.docs = docs or {}

    def add_files(self, files, doc_label_fmt=u'{path}-{basename}', doc_label_path_join='_'):
        for fpath in files:
            with open(fpath) as f:
                text = f.readlines()

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

    def add_folder(self, folder, valid_extensions=('txt',), strip_folderpath_from_doc_label=True,
                   doc_label_fmt=u'{path}-{basename}', doc_label_path_join='_'):
        if type(valid_extensions) is str:
            valid_extensions = (valid_extensions,)

        for root, _, files in os.walk(folder):
            if not files:
                continue

            for fname in files:
                fpath = os.path.join(root, fname)
                with open(fpath) as f:
                    text = f.readlines()

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

    def from_files(self, *args, **kwargs):
        self.docs = {}
        self.add_files(*args, **kwargs)

        return self

    def from_folder(self, *args, **kwargs):
        self.docs = {}
        self.add_folder(*args, **kwargs)

        return self

    def from_pickle(self, picklefile):
        self.docs = unpickle_file(picklefile)

        return self

    def to_pickle(self, picklefile):
        pickle_data(self.docs, picklefile)

    def split_by_paragraphs(self, break_on_num_newlines=2, new_doc_label_fmt=u'{doc}-{parnum}'):
        tmp_docs = {}
        for dl, doc in self.docs.items():
            pars = paragraphs_from_lines(doc, break_on_num_newlines)
            for i, p in enumerate(pars):
                new_dl = new_doc_label_fmt.format(doc=dl, parnum=i+1)
                tmp_docs[new_dl] = p

        assert len(tmp_docs) >= len(self.docs)
        self.docs = tmp_docs


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
    least `break_on_num_newlines` line breaks from another paragraph.
    Return a list of paragraphs, each paragraph containing a string of sentences.
    """
    n_lines = len(lines)
    paragraphs = []
    n_par_newlines = 0
    cur_par = ''
    # iterate through all lines
    for i, l in enumerate(lines):
        if not (l.startswith('\n') or l.startswith('\r\n')):
            n_par_newlines = 0

        if l.endswith('\n'):
            n_par_newlines += 1

        cur_par += ' ' + l.strip()

        # create a new paragraph after at least 2 line breaks or when we reached the end of document
        if n_par_newlines == break_on_num_newlines or i == n_lines - 1:
            paragraphs.append(cur_par)
            cur_par = ''
            n_par_newlines = 0

    return paragraphs
