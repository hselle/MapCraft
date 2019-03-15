# MapCraft
### 3d Map Modeling Tool
![](https://imgur.com/GZpcCxo.png)
### Features:
+ Generate a 3d map from any heightmap image.
+ Navigate this 3d map with the mouse and keyboard.
+ Dynamically generate a navigable world using real topography data from NASA's Shuttle Radar Topography Mission(SRTM)
### Dependencies
Opencv2, Pyglet
### Requirements:
Python 2.7.1, Windows or Linux machine.

## Usage
`python heightmap.py <input.jpg> <output.txt> <max-height> <total-blocks>`
`python mainheightmap.py <output.txt>`
## Package Contents:

**mainheightmap.py:**
Loads elevation data from a heightmap image and renders a 3d navigable map using Pyglet.

**heightmap.py:**
Parses heightmap images using openCV and generates a .txt file which is used to generate the 3d map.

**gradient.png:**
Textures

**usa.jpg, world.jpg, usa.txt, world.txt:**
Two example heightmap images and their corresponding .txt files produced by heightmap.py

**maindynamic:**
Dynamically loads elevation data from SRTM(Shuttle Radar Topography Mission) and renders a 3d navigable world using Pyglet.

**elevation.py:**
Used by maindynamic to get the SRTM elevation data.

------------

By Harrison Selle and Nick Crews
Known Issues: maindynamic not currently functioning due to "bad zip file" error




