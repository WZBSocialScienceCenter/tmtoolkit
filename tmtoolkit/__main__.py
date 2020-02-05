"""
tmtoolkit – Text Mining and Topic Modeling Toolkit for Python

CLI module

Markus Konrad <markus.konrad@wzb.eu>
"""

if __name__ == '__main__':
    import sys
    import subprocess
    import json

    from tmtoolkit.preprocess import DEFAULT_LANGUAGE_MODELS

    def _setup(args):
        try:
            import spacy
            from spacy.cli.download import download
        except ImportError:
            print('error: required package "spacy" is not installed', file=sys.stderr)
            exit(1)

        if not args:
            print('error: you must pass a list of two-letter ISO 639-1 language codes to install the respective '
                  'language models', file=sys.stderr)
            exit(2)
        else:
            install_languages = []
            for arg in args:
                install_languages.extend([l for l in map(str.strip, arg.split(',')) if l])

        print('checking if required spaCy data packages are installed...')

        try:
            piplist_str = subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--disable-pip-version-check',
                                                   '--format', 'json'])
        except subprocess.CalledProcessError as exc:
            print('error: calling pip failed with the following error message\n' + str(exc), file=sys.stderr)
            exit(3)

        piplist = json.loads(piplist_str)
        installed_pkgs = set(item['name'] for item in piplist)
        model_pkgs = dict(zip(DEFAULT_LANGUAGE_MODELS.keys(),
                              map(lambda x: x.replace('_', '-'), DEFAULT_LANGUAGE_MODELS.values())))

        for lang in install_languages:
            if lang not in DEFAULT_LANGUAGE_MODELS.keys():
                print('error: no language model for language code "%s"' % lang, file=sys.stderr)
                exit(4)

            lang_model_pkg = model_pkgs[lang]

            if lang_model_pkg in installed_pkgs:
                print('language model package "%s" for language code "%s" is already installed -- skipping'
                      % (lang_model_pkg, lang))
                continue

            lang_model = DEFAULT_LANGUAGE_MODELS[lang]
            print('installing language model "%s" for language code "%s"...' % (lang_model, lang))
            download(lang_model)

        print('done.')

    commands = {
        'setup': _setup
    }

    if len(sys.argv) <= 1:
        print('available commands: ' + ', '.join(commands.keys()))
        exit(1)

    cmd = sys.argv[1]
    if cmd in commands.keys():
        commands[cmd](sys.argv[2:])
    else:
        print('command not supported:', cmd, file=sys.stderr)
        exit(2)
