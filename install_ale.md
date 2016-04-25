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

Let's try the emulator for running Breakout. 

```bash
curl http://atariage.com/2600/roms/Breakout.zip | tar -xf- -C .
python doc/examples/python_example.py Breakout.bin
```

You should see something like this:

```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  1849  100  1849    0     0   5689      0 --:--:-- --:--:-- --:--:--  5689

A.L.E: Arcade Learning Environment (version 0.5.1)
[Powered by Stella]
Use -help for help screen.
Game console created:
  ROM file:  Breakout.bin
  Cart Name: Breakout - Breakaway IV (1978) (Atari)
  Cart MD5:  f34f08e5eb96e500e851a80be3277a56
  Display Format:  AUTO-DETECT ==> NTSC
  ROM Size:        2048
  Bankswitch Type: AUTO-DETECT ==> 2K

Running ROM file...
Random seed is 123
Episode 0 ended with score: 0
Episode 1 ended with score: 4
Episode 2 ended with score: 0
Episode 3 ended with score: 1
Episode 4 ended with score: 3
Episode 5 ended with score: 2
Episode 6 ended with score: 1
Episode 7 ended with score: 0
Episode 8 ended with score: 1
Episode 9 ended with score: 0
```


## Links

For more about ALE, see its [PDF manual](https://github.com/mgbellemare/Arcade-Learning-Environment/raw/master/doc/manual/manual.pdf) and the [github repo](https://github.com/mgbellemare/Arcade-Learning-Environment).
