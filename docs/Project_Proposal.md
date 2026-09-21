# Project Proposal

**Project Title:** NN-Steiner: A Mixed Neural-Algorithmic Approach for the Rectilinear Steiner Minimum Tree Problem

## Problem Description

### Problem

When designing computer chips in Very Large Scale Integrated Circuits (VLSI Circuits), in some cases, physical design engineers have to connect millions of wires that can only be connected to different points horizontally or vertically. Since there are millions of interconnects (wires) completing such circuits, an efficient way is extremely important as wire length in VLSI design can directly impact key performance metrics like dynamic power, routing congestion, and even timing delay. One possible way for a more efficient result is using the Rectilinear Steiner Minimum Tree (RSMT) method. But the challenge is that RSMT is NP-complete, so finding the exact optimal solution becomes computationally difficult as the number of points starts growing. Existing exact methods (like GeoSteiner) can be expensive, while heuristic or purely neural approaches (like FLUTE: Fast Lookup Table Based Rectilinear Steiner Minimum Tree algorithm) may struggle with scalability or generalization. This paper addresses this problem by proposing NN-Steiner, which combines neural networks with the structure of Arora's approximation algorithm.

### Why It's Important or Interesting

As mentioned earlier, the length of the wires connecting various components in a circuit affects overall chip performance. So minimizing the total wire length directly reduces a chip's power consumption, speeds up signal processing, and also prevents physical congestion or overcrowding on the chip. Incorporating a smart AI approach to improve this can be very useful for future chip layouts, not only in terms of efficiency and time, but also in terms of cost savings and scalability. This paper brings a very interesting approach of creating a hybrid AI system: instead of asking AI to solve the entire problem, the authors created NN-Steiner, which is a hybrid approach. It uses Arora's PTAS algorithm as the mathematical reference to break the big chip into small pieces, and lets small AI algorithms quickly solve those sub-tasks. Now, because the AI only handles small, fixed-sized tasks, one can train it on tiny problem setups, and it will successfully solve massive, complex chip layouts without breaking or overfitting.

### Who Might Benefit from This Solution

The main areas to benefit from this research would be companies involved in large-scale VLSI and electronic design automation (EDA), especially those working on areas like physical design, routing, interconnect planning, and wirelength estimation. Engineers working on AI optimization and research can also benefit from this, as the paper demonstrates how neural networks can be embedded inside bigger algorithms rather than solving difficult optimization problems end-to-end. The paper also suggests that this framework can be generalized to 3D IC designs and to compute obstacle-avoiding RSMTs, which again have important applications in VLSI design.

## Dataset

**Paper Link:** NN-Steiner: A Mixed Neural-Algorithmic Approach for the Rectilinear Steiner Minimum Tree Problem

**Dataset:** NN-Steiner

This paper does not use a fixed public dataset. Instead, the authors generate synthetic point-set data for training and testing.

Some publicly available datasets:
- https://www.ispd.cc/contests/18/
- https://www.ispd.cc/contests/19/

### Dataset Description

The paper uses 120,000 synthetically generated 2D pointsets for training. Each sample consists of point coordinates arranged on a 100 x 100 grid, along with quadtree and portal information. The target labels are derived from exact Steiner-tree solutions generated using GeoSteiner, indicating which portals should be selected as part of the final routing tree. The task is therefore mainly a portal-selection classification problem within a larger neural-algorithmic optimization framework.

### Anticipated Preprocessing

- Normalizing the point coordinates
- Dividing the layouts into quadtree cells
- Padding so that the cells have a fixed input size
- Since the data is generated artificially, no major cleaning should be needed. However, if the public dataset is used instead, it may require cleaning for empty or missing values, plus padding.

## Expected Challenges

There are several potential difficulties we expect to encounter as part of our project. One of them is the large number of non-Steiner points on the edge of each cell. The model may take an easy way out by simply predicting zero for all portals and end up with an artificially low loss. In order to avoid that, we intend to utilize a heavily weighted binary cross-entropy loss function that gives actual Steiner points a much larger weighting (m+1) so that the network is forced to learn about the smaller class. Also, due to the recursive nature of the model's tree-like structure, each input produces a unique quadtree shape. This typically prohibits us from using standard GPU-based batching. We can address both of these issues by generating synthetic data at a fixed depth (e.g., d=3) and then randomly dropping points from leaf cells to ensure consistency in data shapes.

Lastly, finding the right balance for how complex a leaf cell can be will require hyperparameter tuning. More specifically, we will have to determine the maximum number of points that can exist within a single leaf cell. A smaller maximum number of points would likely result in many almost-empty leaves producing very few useful training examples; however, if the maximum number of points is too high, the basic task becomes too difficult for the network to learn. We intend to begin with the published successful baseline parameters, including a maximum of four points per leaf and a portal density of 15, and then fine-tune these values based on how well the model performs during our initial training runs.

## Team Roles

### Data Collection: Jaspreet Aujla

- Research and identify suitable synthetic and public VLSI datasets.
- Generate synthetic point-set data based on the NN-Steiner approach.
- Preprocess and format the data for model training.
- Compare available datasets and determine which is most suitable for the project.

### Modeling: Andrew Shillman

- Study and understand the NN-Steiner model architecture.
- Implement or adapt the NN-Steiner model using the available code.
- Train the model using the prepared dataset.
- Tune model parameters and work on improving model performance.

### Evaluation: Shraddha Debata

- Evaluate the trained model using appropriate performance metrics.
- Analyze wirelength, accuracy, runtime, and scalability.
- Compare model results with baseline or existing approaches.
- Create graphs and tables to present experimental results.

### Report Writing: Jaspreet Aujla, Andrew Shillman, Shraddha Debata

- Combine the work from data collection, modeling, and evaluation into one project report.
- Write the problem statement, motivation, methodology, dataset description, model approach, and evaluation sections.
- Summarize the experimental results using graphs, tables, and key observations.
- Discuss the strengths, limitations, and possible future improvements of the project.
- Review the report together to make sure the writing is clear, consistent, and properly cited/referenced.
- Prepare the final presentation slides and divide the presentation sections among team members.

## References

Kahng, A. B., Nerem, R. R., Wang, Y., & Yang, C.-Y. (2024). NN-Steiner: A mixed neural-algorithmic approach for the rectilinear Steiner minimum tree problem. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(12), 13022-13030.

OpenAI. (2026). ChatGPT [Large language model]. https://chatgpt.com/

## AI Tool Usage

ChatGPT (OpenAI) was used as a learning and writing support tool during the preparation of this proposal. It was used to help understand key concepts related to the paper, including the Rectilinear Steiner Minimum Tree (RSMT) problem, NP-completeness, Arora's PTAS, FLUTE, GeoSteiner, quadtree decomposition, portals, and the NN-Steiner approach. ChatGPT was also used for grammar correction, sentence restructuring, and improving the clarity of the proposal. The technical information and conclusions of the project are based on the original research paper and cited sources rather than solely on AI-generated responses.
