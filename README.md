# Suturis

Real-time image stitching software for video feeds. This software aims to provide a highly configurable interface to realize stiching of video feeds. It has a pipeline like architechture, meaning data comes in via readers, is then processed and output to writers. All steps, the readers, writers and implementations of the stitching are exchangable.

## Usage

Starting the software is relatively straightforward. After cloning, make sure the required packages in the [Pipfile](Pipfile) are installed, either by using pipenv or just user-/system-wide with pip. Consider installing `opencv-python-headless` instead of the default opencv package, if GUI output isn't relevant (like the ScreenOutput writer) to shrinken the package size.
Then create a config just like the [default one](src/config.yaml) to adjust the applications pipeline to your needs. Start the software by executing the [main file](src/main.py) and passing your config as a positional argument (or nothing to use default).
In case the default config is used, make sure the input sources are actually exisiting.
Output data can be found [here](data/out/) and logs [here](log/).

## Structure

The main code base is as usual in the [`src/`](src/) directory with the `main.py` file as the entry point. Right next to it there is the default config, more can be found in the [examples folder](data/examples/).
Defining new readers or writers can be done [here](src/suturis/io/), just create new classes deriving from the respective base class. The same goes for any processing handlers, located in the subdirectories [here](src/suturis/processing/computation/).

One iteration goes as follows: Data is fetched from the readers, then processed, and send to the writers. Note that the processing is obviously the CPU-intensive part, while the reading might be IO-blocking.
The processing itself consists of three steps, namely optional precessing, finding the homography and the mask computation. In the main loop the resulting params of these operations are just applied, while in a seperate process said params are constantly updated.

The [`test/`](test/) directory just contains various experimental snippets, used for performance checks, comparison of algorithms and other things.

## Config

The config features three main sections, the IO part, the stitching part and the logging part. Logging is the easiest to explain, since it just follows the config schema of pythons internal logging module and is essentially just passed on to it.
The IO and stitching sections are quite similar, as they provide information about the classes to create and use. The readers, writers in IO and the preprocessors in stitching accept a list of classes, while homography and masking only allow for one definition. Any definition must include a `type` which is the name of the class, and any other params will be passed to the constructor of that class as keyword arguments. The locations to the classes and their base classes can be found in the [structure section](#structure).
