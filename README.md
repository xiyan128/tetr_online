# Tetr Online: A Guideline Tetris Clone with Rust & Bevy

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Controls](#controls)
4. [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Running the game](#running-the-game)
5. [WebAssembly (WASM) Support](#webassembly-support)
6. [Acknowledgements](#acknowledgements)
7. [Contributing](#contributing)
8. [License](#license)

<img width="1289" alt="Screenshot 2023-04-01 at 3 00 39 AM" src="https://user-images.githubusercontent.com/17198282/229279595-87174f68-c88e-41cd-81c2-47be0f383f72.png">

## Introduction

This project is an open-source implementation of a guideline-compliant Tetris clone, created using the Rust programming language. It is still in early development and borrows some resources from other open-source Tetris implementations, such as sound effects from Techmino, which will be replaced later.

Try an early demo here: https://www.xiyan.dev/tetr_online/

## Features

- SRS (Super Rotation System): A standard rotation system for Tetrimino shapes.
- Bag of 7: A randomization system that ensures each of the 7 Tetrimino shapes will appear once before repeating.
- Hard drop: Instantly drops the Tetrimino to the bottom of the playfield.
- Hold: Allows the player to save a Tetrimino for later use.
- Ghost pieces: A transparent representation of the current Tetrimino, showing where it would land if hard dropped.
- Half-second lock delay: Provides a brief delay before the Tetrimino locks into place, allowing for last-second adjustments.
- Score calculation: Calculates the player's score based on lines cleared and other factors.
- T-spin detection: Recognizes and rewards the player for successfully executing T-spins.

## Controls

- Arrow keys: Move the Tetrimino left, right, or down.
- Z/X/Up arrow: Rotate the Tetrimino.
- Space: Hard drop.
- Left Shift: Hold the current Tetrimino.

## Getting Started

### Prerequisites

To build and run the project, you'll need the following:

1. [Rust programming language](https://www.rust-lang.org/tools/install) installed on your system.
2. A compatible web browser for running the game in WebAssembly (WASM) mode.

### Installation

1. Clone the repository to your local machine:

```
git clone https://github.com/xiyan128/tetr_online.git
```

2. Navigate to the project directory:

```
cd tetr_online
```

### Running the game

1. To run the game locally using the Rust toolchain, execute the following command:

```
cargo run
```

## WebAssembly (WASM) Support

This Tetris clone can be compiled to WebAssembly (WASM) and played in a web browser. Follow the instructions from [Trunk](https://trunkrs.dev/) for guidance.


## Goals and Motivations

This Tetris clone is a hobby project created by a regular Tetris fan. The primary motivation behind the project is to develop a modern, multiplayer Tetris implementation that rivals popular platforms such as Jstris and Tetr.io. The game aims to support both web and native platforms, while delivering better performance.

However, progress on the project is limited by the author's available time. The hope is that by sharing this project with the open-source community, other Tetris enthusiasts and developers can contribute to its development, shaping it into a polished and feature-rich game that serves as a testament to the passion and dedication of Tetris fans everywhere.


## Acknowledgements

- Sound effects: [Techmino](https://github.com/26F-Studio/Techmino)
- Additional open-source Tetris implementations that inspired and informed this project.

Please note that this project is an independently developed, open-source Tetris clone and is not affiliated with or endorsed by The Tetris Company or any other entity owning rights to the Tetris brand or game. The purpose of this project is to create a Tetris-inspired game for educational and recreational purposes.

While the project aims to adhere to Tetris guidelines and implements some Tetris-specific features, it does not use any copyrighted assets or materials from the official Tetris games. Some resources, such as sound effects, have been borrowed from other open-source projects and will be replaced in the future with original assets.

If you are a copyright holder and believe that any content in this project infringes on your copyrights, please contact the project author with the relevant information. Upon receiving a valid notice, the project author will take appropriate action to address any copyright concerns. (This section is entirely GPT-4 generated, you get the idea XP

## Contributing

Contributions are welcome! If you have an idea, bug report, or feature request, please create an issue or submit a pull request.

## License

This project is released under the MIT License.
