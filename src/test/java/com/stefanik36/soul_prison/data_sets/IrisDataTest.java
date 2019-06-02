package com.stefanik36.soul_prison.data_sets;

import com.stefanik36.soul_prison.builder.*;
import com.stefanik36.soul_prison.data.DataResultTuple;
import com.stefanik36.soul_prison.data.TrainData;
import com.stefanik36.soul_prison.model.TestResult;
import com.stefanik36.soul_prison.network_components.layer.Step;
import com.stefanik36.soul_prison.network_components.layer.StepImpl;
import com.stefanik36.soul_prison.network_components.layer.StepInput;
import com.stefanik36.soul_prison.network_components.network.Network;
import com.stefanik36.soul_prison.network_components.neuron.NeuronNode;
import com.stefanik36.soul_prison.source.DataSource;
import com.stefanik36.soul_prison.util.PlotUtil;
import com.stefanik36.soul_prison.util.TestUtil;
import io.vavr.collection.List;
import io.vavr.control.Option;
import junit.framework.TestCase;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class IrisDataTest {


    @Test
    public void irisDataSigmoid() {
        Random random = new Random(66);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(
                ValidationFunctionFactory.classification(),
                4,
                3,
                4, 3
        )
                .setBias(0.0)
                .setLearningRate(0.3)
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid());

        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        TrainData trainData = new TrainData(DataSource.getIrisData(), random, categoryValues).setTest(0.3);

        TestResult result = TestUtil.testDifferentRandoms((td, net) -> {
                    net.train(td, 200);
                    net.test(td);
//                    PlotUtil.plotResult(net.getTrainSummaryList());
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                20
        );

        assertEquals(0.9677777777777777, result.getTestAccuracy(), 0.001);
    }


    @Test
    public void irisDataSigmoidWithBias() {
        Random random = new Random(666);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.classification(), 4, 3, 4, 3)
                .setBias(1.0)
                .setLearningRate(0.3)
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMomentum(0.0);

        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        TrainData trainData = new TrainData(DataSource.getIrisData(), random, categoryValues).setTest(0.3);

        TestResult result = TestUtil.testDifferentRandoms((td, net) -> {
                    net.train(td, 100);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                20
        );


        assertEquals(0.9155555555555557, result.getTestAccuracy(), 0.001);
    }


    @Test
    public void irisDataSigmoidWithBiasAndMomentum() {
        Random random = new Random(666);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.classification(), 4, 3, 4, 3)
                .setBias(1.0)
                .setLearningRate(0.3)
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMomentum(0.3);


        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        TrainData trainData = new TrainData(DataSource.getIrisData(), random, categoryValues).setTest(0.3);


        TestResult result = TestUtil.testDifferentRandoms((td, net) -> {
                    net.train(td, 100);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                20
        );


        assertEquals(0.9400000000000001, result.getTestAccuracy(), 0.001);

    }

    @Test
    public void irisDataSigmoidFullyConnectedByBuilder() {
        Random random = new Random(66);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.classification(), 4, 3, 4, 3)
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMean(0.0)
                .setRange(1.0);

        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        TrainData trainData = new TrainData(DataSource.getIrisData(), random, categoryValues).setTest(0.3);


        TestResult result = TestUtil.testDifferentRandoms((td, net) -> {
                    net.train(td, 200);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                20
        );


        assertEquals(0.9677777777777777, result.getTestAccuracy(), 0.001);
    }


    @Test
    public void irisData753() {
        Random random = new Random(66);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.classification(), 4, 3, 7, 5)
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMean(0.0)
                .setRange(1.0);

        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        TrainData trainData = new TrainData(DataSource.getIrisData(), random, categoryValues).setTest(0.3);


        TestResult result = TestUtil.testDifferentRandoms((td, net) -> {
                    net.train(td, 100);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                20
        );


        assertEquals(0.9611111111111112, result.getTestAccuracy(), 0.001);
    }


    @Test
    public void irisDataSigmoidFullyConnectedByBuilderDeleteConnection() {

        Random random = new Random(666);
        NetworkBuilder networkBuilder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.classification(), 4, 3, 4, 3)
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setOutputResolver(OutputResolverFactory.onlyTestResults())
                .setMean(0.0)
                .setRange(1.0);


        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        TrainData trainData = new TrainData(DataSource.getIrisData(), random, categoryValues).setTest(0.3);
        TestResult result = TestUtil.testDifferentRandoms((td, net) -> {
                    Step output = net.getSteps().last();
                    output.getNeuronsWithBias()
                            .filter(n -> n instanceof NeuronNode)
                            .map(n -> (NeuronNode) n).subSequence(0, 1)
                            .forEach(
                                    n -> net.getAllNeurons()
                                            .forEach(n::removeConnection)
                            );
                    net.train(td, 200);
                    net.test(td);
                    return net.getTestResult();
                },
                networkBuilder,
                trainData,
                random,
                20
        );

        assertEquals(0.6144444444444443, result.getTestAccuracy(), 0.001);
    }

    @Test
    public void irisDataAutoEncoderFreezeWeightsAddedToNetwork02() {
        Random random = new Random(44);
        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        List<DataResultTuple> iData = DataSource.getIrisData();
        TrainData trainData = new TrainData(iData, random, categoryValues);
        Network autoEncoder = NetworkBuilder.initFullyConnected(
                ValidationFunctionFactory.equalWithAccuracy(0.05),
                4,
                4,
                3)
                .setBias(0.0)
                .setLearningRate(0.3)
                .setMomentum(0.1)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setMean(0.0)
                .setRange(1.0)
                .setRandom(random)
                .build();


        autoEncoder.trainAutoEncoder(trainData, 1000);
        autoEncoder.testAutoEncoder(trainData);
        TestResult etr = autoEncoder.getTestResult();
        TestCase.assertEquals(0.6, etr.getTestAccuracy(), 0.001);
//        PlotUtil.plotResult(autoEncoder.getTrainSummaryList(), etr);


        StepInput input = StepBuilder.initInput(4)
                .setRandom(random)
                .setName("input01")
                .buildInput();

        StepImpl s02 = StepBuilder.updatePrevNeurons((StepImpl) autoEncoder.getSteps().get(1), input.getNeuronsWithBias())
                .setRandom(random)
                .buildWithConnections();
        s02.setFrozenWeights(true);

        StepImpl s03 = StepBuilder.initFullyConnected(3, input.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("s03")
                .buildWithConnections();

        StepImpl s04 = StepBuilder.initFullyConnected(3, s02.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("s04")
                .buildWithConnections();

        StepImpl s05 = StepBuilder.initFullyConnected(3, s03.getNeuronsWithBias().appendAll(s04.getNeurons()), Option.none())
                .setRandom(random)
                .setName("s05")
                .buildWithConnections();

        Network network = NetworkBuilder.initFromSteps(ValidationFunctionFactory.binary(), input, s02, s03, s04, s05).build();

        network.train(trainData, 1000);
        network.test(trainData);
        TestResult tr = network.getTestResult();

//        System.out.println(etr.getTestAccuracy() + " " + tr.getTestAccuracy());

        TestCase.assertEquals(0.9555555555555556, tr.getTestAccuracy(), 0.001);
//        PlotUtil.plotResult(network.getTrainSummaryList(), tr);

    }


}
