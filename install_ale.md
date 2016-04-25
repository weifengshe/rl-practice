# Installation instructions for Arcade Learning Environment

In some of the exercises we are going to use [Arcade Learning Environment](http://www.arcadelearningenvironment.org/) (ALE) to run old Atari 2600 games.

## OS X

To install ALE on OS X, make sure you have [Homebrew](http://brew.sh/) installed. Then run the following:

```bash
brew install cmake
brew install sdl
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment/
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
make -j 4
sudo pip install .
```

## Linux

To install ALE on Linux, run the following:

```bash
sudo apt-get install cmake libsdl1.2-dev
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment/
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
make -j 4
sudo pip install .
```

## Testing the installation

You can test if ALE works by running:

```bash
python doc/examples/python_example.py
```

## Links

For more about ALE, see its [PDF manual](https://github.com/mgbellemare/Arcade-Learning-Environment/raw/master/doc/manual/manual.pdf) and the [github repo](https://github.com/mgbellemare/Arcade-Learning-Environment).
