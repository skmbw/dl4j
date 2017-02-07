package com.vteba.dl4j;

import com.vteba.utils.common.PathUtils;
import org.apache.log4j.PropertyConfigurator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * deeplearning4j学习
 *
 * @author yinlei
 * @since 2016/11/21.
 */
public class DL4JTest {

    static {
        String path = PathUtils.getClassPath(DL4JTest.class);
        PropertyConfigurator.configure(path + "log4j.xml");
    }

    private static final Logger LOGGER = LoggerFactory.getLogger(DL4JTest.class);

    public static void main(String[] args) throws Exception {
        int nChannels = 1;  //通道数目
        int outputNum = 10; //输出层神经元数目
        int batchSize = 1000; //batch的大小 图像块
        int nEpochs = 10; // epoch数目
        int iterations = 1; //迭代次数
        int seed = 123; // 随机数

        LOGGER.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        /*
         * 在使用的时候需要事先创建DataSetIterator，一般是相当于自己集成吧？
         */
        /**
         System.out.println("Total examples in the iterator : " + mnistTrain.totalExamples());// 60000个例子
         System.out.println("Input columns for the dataset " + mnistTrain.inputColumns());// 28*28 每一个图像大小
         System.out.println("The number of labels for the dataset : " + mnistTrain.totalOutcomes()); // 一共10类
         System.out.println("Batch size: "+mnistTrain.batch());// 每一次批次训练的输入样本数目
         **/
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345);
        LOGGER.info("Build model....");
        // 配置神经网络
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)// 使用L2正则化
                .learningRate(0.01) // 学习步长
                .weightInit(WeightInit.XAVIER) //权值初始化
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //采用随机梯度下降法
                .updater(Updater.NESTEROVS).momentum(0.9) //全职更新方式
                .list() //神经网络非输入层的层数
                .layer(0, new ConvolutionLayer.Builder(2, 5) // 卷积层 5*5， 这里可以自己进行
                        .nIn(nChannels) // 通道数
                        .stride(1, 1) // 卷积神经网络进行卷积时的步长
                        .nOut(20).dropOut(0.5) // gropOut 是规定该层神经元激活的数目0.5
                        .activation("relu") // 激活函数
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // 采样层
                        .kernelSize(2, 2) // 核大小 2*2
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder().activation("relu")  //全连阶层，稠密层数
                        .nOut(500).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax") //激活函数
                        .build())
                .backprop(true) //是否支持后向传播
                .pretrain(false) //是否预先训练
                .setInputType(InputType.convolutionalFlat(28, 28, 1)); // 构建多层感知机
//         new ConvolutionLayerSetup(builder,28,28,1);//构建多层感知机

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init(); //模型初始化

        LOGGER.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for (int i = 0; i < nEpochs; i++) {
            model.fit(mnistTrain);
            LOGGER.info("*** Completed epoch {} ***", i);
            LOGGER.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while (mnistTest.hasNext()) {
                DataSet ds = mnistTest.next();
                INDArray output = model.output(ds.getFeatureMatrix());
                eval.eval(ds.getLabels(), output);
            }
            LOGGER.info(eval.stats());
            mnistTest.reset();
            // 将训练好的神经网络
        }
        LOGGER.info("****************Example finished********************");
    }
}
