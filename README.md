# frontier_explorer

ROS2 c++ package for frontier-based exploration.

## Motivation

explore_lite, a common implementation of frontier-based exploration, makes several limiting assumptions. First of all, it assumes it makes sense to use the centroid of a frontier region as a goal. What if we are beginning with no knowledge of the map and with a 2-d lidar? We are likely to be at the centroid of a donut-shaped frontier - this amounts to no motion and - exit of exploration! This is clearly not viable.

## frontier exploration

See ```src/frontier_explorer.cpp```.

This node was developed by attempting to inform GPT5 of my specifications and letting GPT5 write the node. This took some editing after the fact to debug and clean up the code. My preference would be that all processing is done via an image processing library (for clarity, if nothing else). But, coding with GPT5 is an experiment.

## launch/run

```ros2 run frontier_explorer frontier_explorer_node```

## Work in progress!
Note this is a work in progress.

## Credits

This code was developed and debugged and optimized via close collaboration with GPT5 (OpenAI, August 2025). Getting good code out of GPT5 is a ... learning process!