"""
tmtoolkit – Text Mining and Topic Modeling Toolkit for Python

CLI module

Markus Konrad <markus.konrad@wzb.eu>
"""

if __name__ == '__main__':
    import sys

    def _setup(args):
        try:
            import nltk
        except ImportError:
            print('error: required package "nltk" is not installed', file=sys.stderr)
            exit(1)

        if args:
            target = args[0]
        else:
            target = None

        print('checking if required NLTK data packages are installed...')

        if target:
            print('target is', target)

        dl = nltk.downloader.Downloader()

        required = [
            'averaged_perceptron_tagger',
            'punkt',
            'stopwords',
            'wordnet',
            'wordnet_ic'
        ]

        if target == 'test':
            required.append('gutenberg')

        pkgs = dl.packages()

        for p in pkgs:
            if p.id in required:
                print('>', p.id, end=': ')
                if dl.status(p) == dl.NOT_INSTALLED:
                    print('not installed; will try to install now')
                    dl.download(p)
                elif dl.is_stale(p):
                    print('out of date; will try to update now')
                    dl.update(p)
                else:
                    print('ok')

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
