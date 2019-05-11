package com.stefanik36.soul_prison.structures;

import com.stefanik36.soul_prison.builder.ActivationFunctionFactory;
import com.stefanik36.soul_prison.builder.NetworkBuilder;
import com.stefanik36.soul_prison.builder.StepBuilder;
import com.stefanik36.soul_prison.builder.ValidationFunctionFactory;
import com.stefanik36.soul_prison.data.DataResultTuple;
import com.stefanik36.soul_prison.data.InOut;
import com.stefanik36.soul_prison.data.Input;
import com.stefanik36.soul_prison.data.TrainData;
import com.stefanik36.soul_prison.model.TestResult;
import com.stefanik36.soul_prison.network_components.layer.StepImpl;
import com.stefanik36.soul_prison.network_components.layer.StepInput;
import com.stefanik36.soul_prison.network_components.network.Network;
import com.stefanik36.soul_prison.source.DataSource;
import io.vavr.collection.List;
import io.vavr.control.Option;
import org.junit.Test;

import java.util.Random;

import static junit.framework.TestCase.assertEquals;

public class AutoEncoderTest {

    @Test
    public void xorAutoEncoderNonFullyConnected01() {
        Random random = new Random(666);

        StepInput input = StepBuilder.initInput(3)
                .setRandom(random)
                .setName("input01")
                .buildInput();

        StepImpl s02 = StepBuilder.initFullyConnected(2, input.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("s02")
                .buildWithConnections();

        StepImpl s03 = StepBuilder.initFullyConnected(1, input.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("s03")
                .buildWithConnections();

        StepImpl s04 = StepBuilder.initFullyConnected(2, s02.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("s04")
                .buildWithConnections();

        StepImpl s05 = StepBuilder.initFullyConnected(3, s03.getNeuronsWithBias().appendAll(s04.getNeurons()), Option.none())
                .setRandom(random)
                .setName("s05")
                .buildWithConnections();

        Network network = NetworkBuilder.initFromSteps(ValidationFunctionFactory.binary(), input, s02, s03, s04, s05).build();

        List<InOut> iData = List.of(
                new InOut(new Input(List.of(1.0, 1.0, 1.0)), new Input(List.of(1.0, 1.0, 1.0))),
                new InOut(new Input(List.of(1.0, 1.0, 0.0)), new Input(List.of(1.0, 1.0, 0.0))),
                new InOut(new Input(List.of(1.0, 0.0, 1.0)), new Input(List.of(1.0, 0.0, 1.0))),
                new InOut(new Input(List.of(1.0, 0.0, 0.0)), new Input(List.of(1.0, 0.0, 0.0))),
                new InOut(new Input(List.of(1.0, 1.0, 1.0)), new Input(List.of(1.0, 1.0, 1.0))),
                new InOut(new Input(List.of(1.0, 1.0, 0.0)), new Input(List.of(1.0, 1.0, 0.0))),
                new InOut(new Input(List.of(1.0, 0.0, 1.0)), new Input(List.of(1.0, 0.0, 1.0))),
                new InOut(new Input(List.of(1.0, 0.0, 0.0)), new Input(List.of(1.0, 0.0, 0.0)))
        );

        TrainData trainData = new TrainData(iData, random)
                .setValidation(0.0);

        int epochs = 200;
        network.train(trainData, epochs);


        network.test(trainData);
        TestResult tr = network.getTestResult();

        assertEquals(1.0, tr.getTestAccuracy(), 0.001);
    }

    @Test
    public void xorAutoEncoderFullyConnected01() {
        Random random = new Random(666);
        Network network = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.binary(), 3, 3, 2)
                .setBias(0.0)
                .setLearningRate(0.8)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setMean(0.0)
                .setRange(1.0)
                .setRandom(random)
                .build();


        List<InOut> ioData = List.of(
                new InOut(new Input(List.of(1.0, 1.0, 1.0)), new Input(List.of(1.0, 1.0, 1.0))),
                new InOut(new Input(List.of(1.0, 1.0, 0.0)), new Input(List.of(1.0, 1.0, 0.0))),
                new InOut(new Input(List.of(1.0, 0.0, 1.0)), new Input(List.of(1.0, 0.0, 1.0))),
                new InOut(new Input(List.of(1.0, 0.0, 0.0)), new Input(List.of(1.0, 0.0, 0.0))),
                new InOut(new Input(List.of(1.0, 1.0, 1.0)), new Input(List.of(1.0, 1.0, 1.0))),
                new InOut(new Input(List.of(1.0, 1.0, 0.0)), new Input(List.of(1.0, 1.0, 0.0))),
                new InOut(new Input(List.of(1.0, 0.0, 1.0)), new Input(List.of(1.0, 0.0, 1.0))),
                new InOut(new Input(List.of(1.0, 0.0, 0.0)), new Input(List.of(1.0, 0.0, 0.0)))
        );

        TrainData trainData = new TrainData(ioData, random)
                .setValidation(0.0);

        int epochs = 50;
        network.train(trainData, epochs);
        network.test(trainData);
        TestResult tr = network.getTestResult();

        assertEquals(1.0, tr.getTestAccuracy(), 0.001);
    }


    @Test
    public void xorAutoEncoderNormalData01() {
        Random random = new Random(666);
        Network autoEncoder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.binary(), 3, 3, 2)
                .setBias(0.0)
                .setLearningRate(0.8)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setMean(0.0)
                .setRange(1.0)
                .setRandom(random)
                .build();

        List<DataResultTuple> iData = List.of(
                new DataResultTuple(0.0, List.of(1.0, 1.0, 1.0)),
                new DataResultTuple(1.0, List.of(1.0, 1.0, 0.0)),
                new DataResultTuple(1.0, List.of(1.0, 0.0, 1.0)),
                new DataResultTuple(0.0, List.of(1.0, 0.0, 0.0)),
                new DataResultTuple(0.0, List.of(1.0, 1.0, 1.0)),
                new DataResultTuple(1.0, List.of(1.0, 1.0, 0.0)),
                new DataResultTuple(1.0, List.of(1.0, 0.0, 1.0)),
                new DataResultTuple(0.0, List.of(1.0, 0.0, 0.0))
        );


        List<InOut> ioData = iData.map(dt -> new InOut(new Input(dt.getData()), new Input(dt.getClassificationResult(List.of(0.0, 1.0)))));

        TrainData trainData = new TrainData(ioData, random)
                .setValidation(0.0);

        int epochs = 50;
        autoEncoder.trainAutoEncoder(trainData, epochs);
        autoEncoder.testAutoEncoder(trainData);
        TestResult tr = autoEncoder.getTestResult();


        assertEquals(1.0, tr.getTestAccuracy(), 0.001);
    }

    @Test
    public void xorAutoEncoderAddedToNetwork01() {
        Random random = new Random(66);
        Network autoEncoder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.binary(), 3, 3, 2)
                .setBias(0.0)
                .setLearningRate(0.8)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setMean(0.0)
                .setRange(1.0)
                .setRandom(random)
                .build();

        List<DataResultTuple> iData = List.of(
                new DataResultTuple(0.0, List.of(1.0, 1.0, 1.0)),
                new DataResultTuple(1.0, List.of(1.0, 1.0, 0.0)),
                new DataResultTuple(1.0, List.of(1.0, 0.0, 1.0)),
                new DataResultTuple(0.0, List.of(1.0, 0.0, 0.0)),
                new DataResultTuple(0.0, List.of(1.0, 1.0, 1.0)),
                new DataResultTuple(1.0, List.of(1.0, 1.0, 0.0)),
                new DataResultTuple(1.0, List.of(1.0, 0.0, 1.0)),
                new DataResultTuple(0.0, List.of(1.0, 0.0, 0.0))
        );


        List<InOut> ioData = iData.map(dt -> new InOut(new Input(dt.getData()), new Input(dt.getClassificationResult(List.of(0.0, 1.0)))));

        TrainData trainData = new TrainData(ioData, random)
                .setValidation(0.0);

        int epochs = 100;
        autoEncoder.trainAutoEncoder(trainData, epochs);
        autoEncoder.testAutoEncoder(trainData);
        TestResult etr = autoEncoder.getTestResult();

        assertEquals(1.0, etr.getTestAccuracy(), 0.001);


        StepInput input = StepBuilder.initInput(3)
                .setRandom(random)
                .setName("input01")
                .buildInput();

        StepImpl s02 = StepBuilder.updatePrevNeurons((StepImpl) autoEncoder.getSteps().get(1), input.getNeuronsWithBias())
                .setRandom(random)
                .buildWithConnections();

        StepImpl s03 = StepBuilder.initFullyConnected(1, input.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("s03")
                .buildWithConnections();

        StepImpl s04 = StepBuilder.initFullyConnected(2, s02.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("s04")
                .buildWithConnections();

        StepImpl s05 = StepBuilder.initFullyConnected(2, s03.getNeuronsWithBias().appendAll(s04.getNeurons()), Option.none())
                .setRandom(random)
                .setName("s05")
                .buildWithConnections();

        Network network = NetworkBuilder.initFromSteps(ValidationFunctionFactory.binary(), input, s02, s03, s04, s05).build();

        network.train(trainData, 250);
        network.test(trainData);
        TestResult tr = network.getTestResult();


        assertEquals(1.0, tr.getTestAccuracy(), 0.001);

    }


    @Test
    public void xorAutoEncoderFreezeWeightsAddedToNetwork() {
        Random random = new Random(66);
        Network autoEncoder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.binary(), 3, 3, 2)
                .setBias(0.0)
                .setLearningRate(0.8)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setMean(0.0)
                .setRange(1.0)
                .setRandom(random)
                .build();

        List<DataResultTuple> iData = List.of(
                new DataResultTuple(0.0, List.of(1.0, 1.0, 1.0)),
                new DataResultTuple(1.0, List.of(1.0, 1.0, 0.0)),
                new DataResultTuple(1.0, List.of(1.0, 0.0, 1.0)),
                new DataResultTuple(0.0, List.of(1.0, 0.0, 0.0)),
                new DataResultTuple(0.0, List.of(1.0, 1.0, 1.0)),
                new DataResultTuple(1.0, List.of(1.0, 1.0, 0.0)),
                new DataResultTuple(1.0, List.of(1.0, 0.0, 1.0)),
                new DataResultTuple(0.0, List.of(1.0, 0.0, 0.0))
        );


        List<InOut> ioData = iData.map(dt -> new InOut(new Input(dt.getData()), new Input(dt.getClassificationResult(List.of(0.0, 1.0)))));

        TrainData trainData = new TrainData(ioData, random)
                .setValidation(0.0);

        int epochs = 100;
        autoEncoder.trainAutoEncoder(trainData, epochs);
        autoEncoder.testAutoEncoder(trainData);
        TestResult etr = autoEncoder.getTestResult();

        assertEquals(1.0, etr.getTestAccuracy(), 0.001);


        StepInput input = StepBuilder.initInput(3)
                .setRandom(random)
                .setName("input01")
                .buildInput();

        StepImpl s02 = StepBuilder.updatePrevNeurons((StepImpl) autoEncoder.getSteps().get(1), input.getNeuronsWithBias())
                .setRandom(random)
                .buildWithConnections();
        s02.setFrozenWeights(true);

        StepImpl s03 = StepBuilder.initFullyConnected(1, input.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("s03")
                .buildWithConnections();

        StepImpl s04 = StepBuilder.initFullyConnected(2, s02.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("s04")
                .buildWithConnections();

        StepImpl s05 = StepBuilder.initFullyConnected(2, s03.getNeuronsWithBias().appendAll(s04.getNeurons()), Option.none())
                .setRandom(random)
                .setName("s05")
                .buildWithConnections();

        Network network = NetworkBuilder.initFromSteps(ValidationFunctionFactory.binary(), input, s02, s03, s04, s05).build();

        network.train(trainData, 250);
        network.test(trainData);
        TestResult tr = network.getTestResult();


        assertEquals(1.0, tr.getTestAccuracy(), 0.001);

    }


    @Test
    public void wineDataAutoEncoder01() {
        Random random = new Random(66);
        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);

        List<DataResultTuple> iData = DataSource.getWineData();

        TrainData trainData = new TrainData(iData, random, categoryValues).setValidation(0.1);

        Network autoEncoder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.equalWithAccuracy(0.01), 13, 13, 10)
                .setBias(0.0)
                .setLearningRate(0.8)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setMean(0.0)
                .setRange(1.0)
                .setRandom(random)
                .build();


        int epochs = 1000;
        autoEncoder.trainAutoEncoder(trainData, epochs);
        autoEncoder.testAutoEncoder(trainData);
        TestResult etr = autoEncoder.getTestResult();
        assertEquals(0.88679, etr.getTestAccuracy(), 0.001);
//        PlotUtil.plotResult(autoEncoder.getTrainSummaryList(), etr);
    }

    @Test
    public void wineDataAutoEncoderFreezeWeightsAddedToNetwork01() {
        Random random = new Random(66);
        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);

        List<DataResultTuple> iData = DataSource.getWineData();

        TrainData trainData = new TrainData(iData, random, categoryValues).setValidation(0.1);

        Network autoEncoder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.equalWithAccuracy(0.01), 13, 13, 10)
                .setBias(0.0)
                .setLearningRate(0.8)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setMean(0.0)
                .setRange(1.0)
                .setRandom(random)
                .build();


        int epochs = 1000;
        autoEncoder.trainAutoEncoder(trainData, epochs);
        autoEncoder.testAutoEncoder(trainData);
        TestResult etr = autoEncoder.getTestResult();
        assertEquals(0.88679, etr.getTestAccuracy(), 0.001);
//        PlotUtil.plotResult(autoEncoder.getTrainSummaryList(), etr);


        StepInput input = StepBuilder.initInput(13)
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

        network.train(trainData, 10000);
        network.test(trainData);
        TestResult tr = network.getTestResult();


        assertEquals(0.75471, tr.getTestAccuracy(), 0.001);
//        PlotUtil.plotResult(network.getTrainSummaryList(), tr);


    }


    @Test
    public void wineDataAutoEncoderFreezeWeightsAddedToNetwork02() {
        Random random = new Random(66);
        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);

        List<DataResultTuple> iData = DataSource.getWineData();

        TrainData trainData = new TrainData(iData, random, categoryValues).setValidation(0.1);

        Network autoEncoder = NetworkBuilder.initFullyConnected(ValidationFunctionFactory.equalWithAccuracy(0.01), 13, 13, 10)
                .setBias(0.0)
                .setLearningRate(0.8)
                .setMomentum(0.0)
                .setActivationFunction(ActivationFunctionFactory.sigmoid())
                .setMean(0.0)
                .setRange(1.0)
                .setRandom(random)
                .build();


//        int epochs = 1000;
//        autoEncoder.trainAutoEncoder(trainData, epochs);
//        autoEncoder.testAutoEncoder(trainData);
//        TestResult etr = autoEncoder.getTestResult();
//        assertEquals(0.88679, etr.getTestAccuracy(), 0.001);
//        PlotUtil.plotResult(autoEncoder.getTrainSummaryList(), etr);


        StepInput input = StepBuilder.initInput(13)
                .setRandom(random)
                .setName("input01")
                .buildInput();

        StepImpl s02 = StepBuilder.updatePrevNeurons((StepImpl) autoEncoder.getSteps().get(1), input.getNeuronsWithBias())
                .setRandom(random)
                .buildWithConnections();
        s02.setFrozenWeights(true);
//        StepImpl s02 = StepBuilder.initFullyConnected(10, input.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
//                .setRandom(random)
//                .setName("s02")
//                .buildWithConnections();

        StepImpl s03 = StepBuilder.initFullyConnected(2, input.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("s03")
                .buildWithConnections();

        StepImpl s05 = StepBuilder.initFullyConnected(3, s03.getNeuronsWithBias().appendAll(s02.getNeurons()), Option.none())
                .setRandom(random)
                .setName("s05")
                .buildWithConnections();

        Network network = NetworkBuilder.initFromSteps(ValidationFunctionFactory.binary(), input, s02, s03, s05).build();

        network.train(trainData, 10000);
        network.test(trainData);
        TestResult tr = network.getTestResult();


        assertEquals(0.73584, tr.getTestAccuracy(), 0.001);
//        PlotUtil.plotResult(network.getTrainSummaryList(), tr);


    }


}
