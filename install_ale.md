# Installation instructions for Arcade Learning Environment

In some of the exercises we are going to use [Arcade Learning Environment](http://www.arcadelearningenvironment.org/) (ALE) to run old Atari 2600 games.

To install ALE, run the following:

```bash
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment/
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
make -j 4
sudo pip install .
```

That's it! You can test if it works by running:

```bash
python doc/examples/python_example.py
```

For more about ALE, see its [PDF manual](https://github.com/mgbellemare/Arcade-Learning-Environment/raw/master/doc/manual/manual.pdf) and the [github repo](https://github.com/mgbellemare/Arcade-Learning-Environment).
