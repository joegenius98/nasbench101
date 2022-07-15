import config
from absl import app

def main(argv):
    print(argv)
    print(config.build_config())

if __name__ == "__main__":
    app.run(main)