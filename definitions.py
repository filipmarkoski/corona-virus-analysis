import datetime
import os


def timer(func):
    def f(*args, **kwargs):
        start = datetime.datetime.now()
        timestamp = '{date:%Y-%m-%d %H:%M:%S}'.format(date=start)
        print(f'Execution started: {timestamp}')
        rv = func(*args, **kwargs)
        end = datetime.datetime.now()
        timestamp = '{date:%Y-%m-%d %H:%M:%S}'.format(date=end)
        print(f'Execution ended: {timestamp}')
        print('Execution elapsed: ', end - start)
        return rv

    return f


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DOCS_DIR = os.path.join(ROOT_DIR, 'docs')
PAPERS_DIR = os.path.join(DOCS_DIR, 'papers')
IMAGES_DIR = os.path.join(DOCS_DIR, 'images')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
TESTS_DIR = os.path.join(ROOT_DIR, 'tests')
PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
LEXICONS_DIR = os.path.join(ROOT_DIR, 'lexicons')
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, 'embeddings')
SERIALIZED_DIR = os.path.join(ROOT_DIR, 'serialized')
SERIALIZED_MODELS_DIR = os.path.join(SERIALIZED_DIR, 'models')

RNG_SEED = 47
log_file = f'{LOGS_DIR}/log.log'


# jupyter notebook set-up

# pip install -U jupyter
# pip install ipykernel && python -m ipykernel install --user --name=.env
# pip install -U spacy && python -m spacy download en_core_web_sm
# python -m nltk.downloader punkt

## pip install -U jupyter && pip install ipykernel && python -m ipykernel install --user --name=.env

def initialize_dirs():
    dir_paths = [DATA_DIR,
                 DOCS_DIR, PAPERS_DIR, IMAGES_DIR,
                 LOGS_DIR,
                 TESTS_DIR,
                 PLOTS_DIR,
                 MODELS_DIR,
                 LEXICONS_DIR,
                 EMBEDDINGS_DIR,
                 SERIALIZED_DIR, SERIALIZED_MODELS_DIR,
                 OUTPUT_DIR]

    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            print(f'{dir_path} created.')

            pkg_init_file = os.path.join(dir_path, '__init__.py')
            if not os.path.exists(pkg_init_file):
                print(f'{pkg_init_file} created.')
                with open(pkg_init_file, 'w') as init_file:
                    init_file.write(f'# Project Name: {ROOT_DIR}')

        else:
            print(f'{dir_path} already exists.')


def main():
    initialize_dirs()

    if not os.path.exists('casual.py'):
        print('casual.py created.')
        with open('casual.py', 'w') as readme:
            readme.write(f'# Project Name: {ROOT_DIR}')

    if not os.path.exists('main.py'):
        print('main.py created.')
        with open('main.py', 'w') as readme:
            readme.write(f'# Project Name: {ROOT_DIR}')

    if not os.path.exists('README.md'):
        print('README.md created.')
        with open('README.md', 'w') as readme:
            readme.write(f'Project Name: {ROOT_DIR}')

    index_html = f'{DOCS_DIR}\\index.html'
    if not os.path.exists(index_html):
        print('index.html created.')
        with open(index_html, 'w') as index:
            index.write(f'Project Name: {ROOT_DIR}')

    os.system('pip freeze > requirements.txt')


if __name__ == '__main__':
    main()
