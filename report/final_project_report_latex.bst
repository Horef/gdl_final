\documentclass[11pt, oneside]{article}   	% use "amsart" instead of "article" for AMSLaTeX format
\usepackage{geometry}                		% See geometry.pdf to learn the layout options. There are lots.
\geometry{a4paper}                   		% ... or a4paper or a5paper or ... 
%\geometry{landscape}                		% Activate for rotated page geometry
%\usepackage[parfill]{parskip}    		% Activate to begin paragraphs with an empty line rather than an indent
\usepackage{graphicx}				% Use pdf, png, jpg, or eps§ with pdflatex; use eps in DVI mode
								% TeX will automatically convert eps --> pdf in pdflatex		
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{amsopn}
\usepackage{hyperref}
\usepackage{subcaption}
\usepackage{float}

%SetFonts

%SetFonts


\title{Final Project Report\\Introduction to Geometric Deep Learning}
\author{Anya Sukachev and Sergiy Horef}
%\date{}							% Activate to display a given date or no date

\begin{document}
\maketitle
\section{Task 1: Point Cloud Classification}
Disclaimer: we have tried as hard as we could to have reproducible results, however, there are small things that torch uses for which we are not able to set the seed.\\
All of the results we show here are reproducible up to 0.05 and show small fluctuations to either side.

\subsection{Method and Architecture choice}
We have started by trying to implement the PointNet++ model, the link to which was provided in the task pdf file. We had to make some changes in the code provided there, as some parts plaindly didn't work, and some worked in an incorrect way (for example, the accuracy calculation returned results higher than 1).\\

When we had a working model, we have run a wandb hyperparameter optimization, and have found that the best model had these paramaters: sample of 512 points for each cloud and batches of size 128. (Other parameters can be found in the 'q1\_code.py' file.)\\

After 30 epochs, this model was achieving the following:\\
Train loss: \textbf{0.0149}, Train accuracy: \textbf{0.9947}\\
Test loss: \textbf{1.1218}, Test accuracy: \textbf{0.8271}\\
This test accuracy result is approximately the best that this model can achieve.

Here are the graphs of the loss and accuracy on both train and tests:
\begin{figure}[H]
\centering
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q1/graphs_pnpp/loss}
  \caption{Loss of PointNet++}
\end{subfigure}%
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q1/graphs_pnpp/accuracy}
  \caption{Accuracy of PointNet++}
\end{subfigure}
\caption{Loss and Accuracy of PointNet++}
\end{figure}

We also provide the number of errors per each class, which will be useful to compare this model against our final choice of GBNet.
\begin{table}[H]
  \begin{center}
  \begin{tabular}{|l|c|c|c|c|c|}
  \hline
    class & bathtub & bed & chair & desk & dresser \\
    num of errors & 19 & 11 & 1 & 41 & 19  \\
    \hline
    class & monitor & night stand & sofa & table & toilet \\
    num of errors & 8 & 44 & 6 & 3 & 5 \\
    \hline
  \end{tabular}
  \caption{Number of errors per each class in PointNet++}
  \end{center}
\end{table}
Least erred class - \textbf{chair}. Most erred class - \textbf{night stand}.\\

Examples of point clouds that our model errer/did not err upon:
\begin{figure}[H]
\centering
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q1/graphs_pnpp/least_erred_class_non_error}
  \caption{Correctly classified on the least erred class}
\end{subfigure}%
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q1/graphs_pnpp/least_erred_class_error}
  \caption{Incorrectly classified on the least erred class}
\end{subfigure}
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q1/graphs_pnpp/most_erred_class_non_error}
  \caption{Correctly classified on the most erred class}
\end{subfigure}%
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q1/graphs_pnpp/most_erred_class_error}
  \caption{Incorrectly classified on the most erred class}
\end{subfigure}
\caption{Examples of correctly and incorrectly classified point clouds for the least and most erred classes. For PointNet++}
\end{figure}

These results are not great, so we have search for an additional model to try.\\

We have found a paper presenting a more complex approach than that of PointNet++ called GBNet (Geometric Back-projection Network) - \href{https://arxiv.org/abs/1911.12885}{paper link}. Creators of the paper have also provided the implementation of the model itself, which we have fit to work on our task - \href{https://github.com/ShiQiu0419/GBNet}{original github link}.\\

As far as we understand from a brief familiarization with the paper - the model operates by enriching the geometric information of points in low-level 3D space explicitly while also applying CNN-based structures to learn local geometric context implicitly.\\
At the low level, the model uses the Geometric Point Descriptor module to enhance the representation of scattered point clouds. This module operates by identifying explicit geometric relations among points. For each point $p_i$ in the point cloud, the model searches for its two nearest neighbors $p_{j1}$ and $p_{j2}$, and uses these points to form a triangle. This triangle helps in estimating various geometric properties such as edges and normal vectors.\\
In the high-level space, the model incorporates the Attentional Back-projection Edge Features Module (ABEM). This module is designed to refine the feature learning process by integrating an error-correcting feedback mechanism. The ABEM captures local geometric context and projects it back into the network, allowing for the correction and enhancement of the feature extraction process.\\

\begin{figure}[H]
    \begin{center}
    \includegraphics[width=0.5\linewidth]{../local_and_global}
    \caption{Enriching geometric features. In low-level space, we explicitly estimate
geometric items as prior knowledge for the network e.g., edges (green vectors),
normals (red vectors). In high-level space, we aggregate neighbors (green
points) to implicitly capture both prominent (red points) and fine-grained
geometric features (purple points). (Fig 1 in the paper.)}
    \end{center}
\end{figure}


Here is an image of the full architecture:
\begin{figure}[H]
    \begin{center}
    \includegraphics[width=\linewidth]{../gbnet}
    \caption{The Geometric Point Descriptor offers more low-level geometric clues for subsequent high-level geometric feature learning
in cascaded ABEMs, representing the point features in multiple scales of embedding space by aggregating local context. The CAA module refines the learned
feature map to avoid channel-wise redundancy. Finally, we use the concatenation of max-pooling and average-pooling results, as well as fully connected layers
to regress the class scores. (Figure 2 in the paper)}
    \end{center}
\end{figure}

For this model we had a very constrained set of possible hyperparameters, due to the limited size of the memore available on the GPU. We have started with the parameters chosen for PointNet++, and gradually lowered the batch size, until we got something that worked. The final parameters are: sample of 512 points for each cloud and batches of size 32.

We have run this model for 10 epoch in total, and it produces much better results:\\
Train loss: \textbf{1.0959}, Train accuracy: \textbf{0.9491}, Train Balanced Accuracy: \textbf{0.9176}\\
Test loss: \textbf{1.1174}, Test accuracy: \textbf{0.9218}, Test Balanced Accuracy: \textbf{0.9189}\\
In this model we preset also the balanced accuracy measure, which presents an average accuracy for all classes (that is, it accounts for the fact that not all classes are equally presented in the test set.)

Here are the graphs of the loss and accuracy on both train and tests:
\begin{figure}[H]
\centering
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q1/graphs/loss}
  \caption{Loss of GBNet}
\end{subfigure}%
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q1/graphs/accuracy}
  \caption{Accuracy of GBNet}
\end{subfigure}
\caption{Loss and Accuracy of GBNet}
\end{figure}

We also provide the number of errors per each class:
\begin{table}[H]
  \begin{center}
  \begin{tabular}{|l|c|c|c|c|c|}
  \hline
    class & bathtub & bed & chair & desk & dresser \\
    num of errors & 0 & 2 & 0 & 24 & 19  \\
    \hline
    class & monitor & night stand & sofa & table & toilet \\
    num of errors & 2 & 18 & 3 & 5 & 1 \\
    \hline
  \end{tabular}
  \caption{Number of errors per each class in GBNet}
  \end{center}
\end{table}
Least erred class - \textbf{bathtub}. Most erred class - \textbf{desk}.\\

Examples of point clouds that our model errer/did not err upon:
\begin{figure}[H]
\centering
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q1/graphs/least_erred_class_non_error}
  \caption{Correctly classified on the least erred class}
\end{subfigure}
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q1/graphs/most_erred_class_non_error}
  \caption{Correctly classified on the most erred class}
\end{subfigure}%
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q1/graphs/most_erred_class_error}
  \caption{Incorrectly classified on the most erred class}
\end{subfigure}
\caption{Examples of correctly and incorrectly classified point clouds for the least and most erred classes. For GBNet}
\end{figure}

We can see that our model struggles the most with the desk, dresser and night stand classes. (In the figure above, we can see an example of a misclassification - cloud of a desk is mistaken for a table.)\\
This is due to the simple fact that at this resolution (512 points) all of them look very much alike. Additionally, table and desk name a very similar set of objects, and are therefore hard to distinguish.\\
On the other hand, bathtub is quite different from everything else, and therefore can be easily classified by our model. (It also has much smoother angles, which is one of the things explicitly accounted for in the GBNet.)

\section{Task 2: Graph Classification}
\subsection{Method and Architecture choice}
We have started by trying the GAT model we have seen in class.\\
In order to use this model for graph classification, we have added a readout function (function that takes embeddings of all nodes and creates a unified embedding based on some aggreagation function), and a fully connnected layer which takes the result produced by the readout function and returns a list of probabilities of each class.\\

We have created a variable implementation, such that we could dinamically define the number of GNN layers, number of attention head, readout aggregation function, and the hidden dimention of the fully connected network.\\

Because from our initial runs (with some randomly chosen parameters) we were already able to get to a validation accuracy of 100\%, we did not feel the need to try a different model altogether, and instead used wandb to make hyperparameter optimization.\\

All variables we have used in the optimization can be found in q2\_wandb.py, but in total we have looked at 1800 different combinations of parameters. Later, we chose only the parameter sets which resulted in validation accuracy of 100\% and all of those we sorted by the minimal validation loss, and overall stability.\\
Our final model has the following results on the training and validation sets after 96 epochs of training:\\
Train loss: \textbf{0.3468}, Train accuracy: \textbf{0.9733}\\
Validation loss: \textbf{0.3457}, Validation accuracy: \textbf{1.0000}\\

\begin{figure}[H]
\centering
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q2/graphs/loss}
  \caption{Loss on Train and Validation}
\end{subfigure}%
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\linewidth]{../results/Q2/graphs/accuracy}
  \caption{Accuracy on Train and Validation}
\end{subfigure}
\caption{Plots of the performance of our final model for Task 2}
\end{figure}

\subsection{Discussion of Results}
We have a very small dataset, which includes only 150 training points, 19 validation and 19 testing points.\\
It is relatively easy to get a high accuracy on 19 points in total, and the complexity of the model is expressive enough to achive an almost perfect accuracy on train. On the contrary, we can see small jumps on the graph of the validation accuracy, which result from the fact that even a single mistake is equivalent to around 5.23\% in accuracy.\\

One thing which we would like to point out in addition to everything that we have said before - the training and validation sets have an unequal number of 0s and 1s, with ration of 1s to 0s as approximately 2:1. Therefore we have manually set the weight in the loss function we have used to give twice more weight to the 0 label.

\end{document}  