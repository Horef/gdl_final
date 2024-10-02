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
\subsection{Method and Architecture choice}
We have st

\section{Task 2: Graph Classification}
\section{Method and Architecture choice}
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