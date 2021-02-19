#!/bin/bash

#This script will set up Java, Spark, and pyspark on a fresh Jetstream instance.

#-----------------------------------------------------------------------------#
#                                                                             #
# JetStream-Spark-Install.sh                                                  #
#                                                                             #
# Authors: Michael Wyatt   (mwyatt12@utk.edu)                                 #
#          Danny Rorabaugh (dror@utk.edu)                                     #
#                                                                             #
# Description: This bash script will download and install Java 8,  Spark      #
#              2.3.1, and pyspark for JetStream instances.  It also builds    #
#              the matplotlib font cache for fresh Anaconda installations.    #
#                                                                             #
# Usage:                                                                      #
#        [1] After starting your JetStream instance, run $ezj to setup        #
#            Anaconda and start a Jupyter Notebook.                           #
#        [2] Press Ctrl-C twice to exit the Jupyter Notebook                  #
#        [3] Execute this script with $./JetStream-Spark-Install.sh           #
#                                                                             #
#-----------------------------------------------------------------------------#

# Install java8
curl -o $HOME/java.tgz -L http://javadl.oracle.com/webapps/download/AutoDL?BundleId=234464_96a7b8442fe848ef90c96a2fad6ed6d1
mkdir $HOME/java
tar -xvzf $HOME/java.tgz -C $HOME/java --strip-components=1
rm $HOME/java.tgz

# Add Java directory to PATH
echo "export JAVA_HOME=$HOME/java" >> $HOME/.bashrc
echo "export PATH=$JAVA_HOME/bin/:$PATH" >> $HOME/.bashrc

# Install spark
curl -o $HOME/spark.tgz -L http://apache.cs.utah.edu/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz
mkdir $HOME/spark
tar -xvzf $HOME/spark.tgz -C $HOME/spark --strip-components=1
rm $HOME/spark.tgz

# Add Spark directory to PATH
echo "export SPARK_HOME=$HOME/spark" >> $HOME/.bashrc
echo "export PATH=$SPARK_HOME/bin/:$PATH" >> $HOME/.bashrc

# Install pyspark
conda install -y pyspark
echo "export SPARK_LOCAL_IP=127.0.0.1" >> $HOME/.bashrc

# Preload matplotlib
python -c 'import matplotlib as mpl; mpl.use("agg"); from matplotlib import pyplot as plt; plt.plot([],[]); plt.show()'

# Source .bashrc
. $HOME/.bashrc
