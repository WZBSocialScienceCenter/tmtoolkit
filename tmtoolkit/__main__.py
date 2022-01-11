"""
tmtoolkit – Text Mining and Topic Modeling Toolkit for Python

CLI module

Markus Konrad <markus.konrad@wzb.eu>
"""

HELP_TEXT = """tmtoolkit installation setup

Run

    python -m tmtoolkit setup <LANGUAGES>

to install all necessary language models for languages listed as
comma-separated language ISO codes in <LANGUAGES>. The list must
be specified without spaces. Example:

    python -m tmtoolkit setup en,de,ru

This will install language models for English, German and Russian.
To install all available language models, run:

    python -m tmtoolkit setup all

You can pass two additional arguments:

    --variants=...  sets the model size variants to install; default is --variants=sm,md
    --no-update     if this argument is passed, only models that are not installed so far will be installed
"""

if __name__ == '__main__':
    import sys
    import subprocess
    import json

    try:
        from tmtoolkit.corpus import DEFAULT_LANGUAGE_MODELS
    except ImportError:
        print('error: tmtoolkit is not installed with the dependencies required for text processing; '
              'install tmtoolkit with the [recommended] or [textproc] option', file=sys.stderr)
        exit(1)

    def _setup(args):
        from spacy.cli.download import download

        variants_switch = '--variants='
        i_variants_arg = None
        for i, arg in enumerate(args):
            if arg.startswith(variants_switch):
                i_variants_arg = i
                break

        if i_variants_arg is not None:
            vararg = args.pop(i_variants_arg)
            variants = vararg[len(variants_switch):].split(',')
        else:
            variants = ['sm', 'md']

        try:
            args.remove('--no-update')
            no_update = True
        except ValueError:
            no_update = False

        if not args:
            print('error: you must pass a list of two-letter ISO 639-1 language codes to install the respective '
                  'language models or the string "all" to install all available language models', file=sys.stderr)
            exit(3)

        if args == ['all']:
            install_languages = list(DEFAULT_LANGUAGE_MODELS.keys())
        else:
            install_languages = []
            for arg in args:
                install_languages.extend([l for l in map(str.strip, arg.split(',')) if l])

        print('checking if required spaCy data packages are installed...')

        try:
            piplist_str = subprocess.check_output([sys.executable, '-m', 'pip', 'list',
                                                   '--disable-pip-version-check',
                                                   '--format', 'json'])
        except subprocess.CalledProcessError as exc:
            print('error: calling pip failed with the following error message\n' + str(exc), file=sys.stderr)
            exit(4)

        piplist = json.loads(piplist_str)
        installed_pkgs = set(item['name'] for item in piplist)

        for modelvar in variants:
            model_pkgs = dict(zip(DEFAULT_LANGUAGE_MODELS.keys(),
                                  map(lambda x: x.replace('_', '-') + '-' + modelvar,
                                      DEFAULT_LANGUAGE_MODELS.values())))

            for lang in install_languages:
                if lang not in DEFAULT_LANGUAGE_MODELS.keys():
                    print(f'error: no language model for language code "{lang}"', file=sys.stderr)
                    exit(5)

                lang_model_pkg = model_pkgs[lang]

                if no_update and lang_model_pkg in installed_pkgs:
                    print(f'language model package "{lang_model_pkg}" for language code "{lang}" is already installed '
                          f'-- skipping')
                    continue

                lang_model = DEFAULT_LANGUAGE_MODELS[lang] + '_' + modelvar
                print(f'installing language model "{lang_model}" for language code "{lang}"...')
                download(lang_model)

        print('done.')

    def _help(args):
        print(HELP_TEXT)

    commands = {
        'setup': _setup,
        'help': _help,
    }

    if len(sys.argv) <= 1:
        print('available commands: ' + ', '.join(commands.keys()))
        print('run `python -m tmtoolkit help` for help')
        exit(6)

    cmd = sys.argv[1]
    if cmd in commands.keys():
        commands[cmd](sys.argv[2:])
    else:
        print('command not supported:', cmd, file=sys.stderr)
        exit(7)
