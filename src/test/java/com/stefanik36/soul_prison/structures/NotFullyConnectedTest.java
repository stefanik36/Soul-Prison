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
import org.junit.Ignore;
import org.junit.Test;

import java.util.Random;

import static junit.framework.TestCase.assertEquals;

public class NotFullyConnectedTest {

    @Test
    public void xorTest02() {
        Random random = new Random(66);

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

        StepImpl s05 = StepBuilder.initFullyConnected(2, s03.getNeuronsWithBias().appendAll(s04.getNeurons()), Option.none())
                .setRandom(random)
                .setName("s05")
                .buildWithConnections();

        Network network = NetworkBuilder.initFromSteps(ValidationFunctionFactory.classification(), input, s02, s03, s04, s05).build();

        int epochs = 450;
        network.train(trainData, epochs);


        network.test(trainData);
        TestResult tr = network.getTestResult();

        assertEquals(1.0, tr.getTestAccuracy(), 0.001);
    }


    /**
     * Hierarchical Deep MLP Structure
     */
    @Test
    public void wineDataTest01() {
        Random random = new Random(666);

        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        List<DataResultTuple> iData = DataSource.getWineData();
        TrainData trainData = new TrainData(iData, random, categoryValues).setValidation(0.2);

        StepInput i01 = StepBuilder.initInput(13)
                .setRandom(random)
                .setName("i01")
                .setBias(0.0)
                .buildInput();

        /*
         * Network 01
         */

        StepImpl net01_s02 = StepBuilder.initFullyConnected(5, i01.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("net01_s02")
                .setBias(0.0)
                .buildWithConnections();

        StepImpl net01_s03 = StepBuilder.initFullyConnected(3, net01_s02.getNeuronsWithBias(), Option.none())
                .setRandom(random)
                .setName("net01_s03")
                .setBias(0.0)
                .buildWithConnections();

        Network net01 = NetworkBuilder.initFromSteps(ValidationFunctionFactory.binary(), i01, net01_s02, net01_s03).build();

        /*
         * Network 02
         */
        StepImpl net02_s04 = StepBuilder.initFullyConnected(5, i01.getNeuronsWithBias().appendAll(net01_s03.getNeurons()), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("net02_s04")
                .setBias(0.0)
                .buildWithConnections();

        StepImpl net02_s05 = StepBuilder.initFullyConnected(3, net02_s04.getNeuronsWithBias(), Option.none())
                .setRandom(random)
                .setName("net02_s05")
                .setBias(0.0)
                .buildWithConnections();

        Network net02 = NetworkBuilder.initFromSteps(ValidationFunctionFactory.binary(), i01, net01_s02, net01_s03, net02_s04, net02_s05).build();


        /*
         * Network 02
         */
        StepImpl net03_s06 = StepBuilder.initFullyConnected(5, i01.getNeuronsWithBias().appendAll(net01_s03.getNeurons()).appendAll(net02_s05.getNeurons()), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("net03_s06")
                .setBias(0.0)
                .buildWithConnections();

        StepImpl net03_s07 = StepBuilder.initFullyConnected(3, net03_s06.getNeuronsWithBias(), Option.none())
                .setRandom(random)
                .setName("net03_s07")
                .setBias(0.0)
                .buildWithConnections();

        Network net03 = NetworkBuilder.initFromSteps(ValidationFunctionFactory.binary(), i01, net01_s02, net01_s03, net02_s04, net02_s05, net03_s06, net03_s07).build();


        net01.train(trainData, 1000);
        net01.test(trainData);
        TestResult new01_tr = net01.getTestResult();

        net02.train(trainData, 1000);
        net02.test(trainData);
        TestResult new02_tr = net02.getTestResult();

        net03.train(trainData, 1000);
        net03.test(trainData);
        TestResult new03_tr = net03.getTestResult();

        System.out.println(
                " n1: " + new01_tr.getTestAccuracy()
                        + " n2: " + new02_tr.getTestAccuracy()
                        + " n3: " + new03_tr.getTestAccuracy()
        );

        assertEquals(0.5849056603773585, new01_tr.getTestAccuracy(), 0.001);
        assertEquals(0.7547169811320755, new02_tr.getTestAccuracy(), 0.001);
        assertEquals(0.8867924528301887, new03_tr.getTestAccuracy(), 0.001);
    }

    @Test
    public void wineDataTestFreeze02() {
        Random random = new Random(666);

        List<Double> categoryValues = List.of(1.0, 2.0, 3.0);
        List<DataResultTuple> iData = DataSource.getWineData();
        TrainData trainData = new TrainData(iData, random, categoryValues).setValidation(0.2);

        StepInput i01 = StepBuilder.initInput(13)
                .setRandom(random)
                .setName("i01")
                .setBias(0.0)
                .buildInput();

        /*
         * Network 01
         */

        StepImpl net01_s02 = StepBuilder.initFullyConnected(10, i01.getNeuronsWithBias(), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("net01_s02")
                .setBias(0.0)
                .buildWithConnections();

        StepImpl net01_s03 = StepBuilder.initFullyConnected(3, net01_s02.getNeuronsWithBias(), Option.none())
                .setRandom(random)
                .setName("net01_s03")
                .setBias(0.0)
                .buildWithConnections();

        Network net01 = NetworkBuilder.initFromSteps(ValidationFunctionFactory.binary(), i01, net01_s02, net01_s03).build();

        /*
         * Network 02
         */
        StepImpl net02_s04 = StepBuilder.initFullyConnected(10, i01.getNeuronsWithBias().appendAll(net01_s03.getNeurons()), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("net02_s04")
                .setBias(0.0)
                .buildWithConnections();

        StepImpl net02_s05 = StepBuilder.initFullyConnected(3, net02_s04.getNeuronsWithBias(), Option.none())
                .setRandom(random)
                .setName("net02_s05")
                .setBias(0.0)
                .buildWithConnections();

        Network net02 = NetworkBuilder.initFromSteps(ValidationFunctionFactory.binary(), i01, net01_s02, net01_s03, net02_s04, net02_s05).build();


        /*
         * Network 02
         */
        StepImpl net03_s06 = StepBuilder.initFullyConnected(10, i01.getNeuronsWithBias().appendAll(net01_s03.getNeurons()).appendAll(net02_s05.getNeurons()), Option.of(ActivationFunctionFactory.sigmoid()))
                .setRandom(random)
                .setName("net03_s06")
                .setBias(0.0)
                .buildWithConnections();

        StepImpl net03_s07 = StepBuilder.initFullyConnected(3, net03_s06.getNeuronsWithBias(), Option.none())
                .setRandom(random)
                .setName("net03_s07")
                .setBias(0.0)
                .buildWithConnections();

        Network net03 = NetworkBuilder.initFromSteps(ValidationFunctionFactory.binary(), i01, net01_s02, net01_s03, net02_s04, net02_s05, net03_s06, net03_s07).build();


        int epochs = 100;

        net01.train(trainData, epochs);

        net01.test(trainData);
        TestResult new01_tr = net01.getTestResult();

        net01_s02.setFrozenWeights(true);
//        net01_s03.setFrozenWeights(true);

        System.out.println(
                " n1: " + new01_tr.getTestAccuracy()
        );


        net02.train(trainData, epochs);
        net02.test(trainData);
        TestResult new02_tr = net02.getTestResult();

        net01.test(trainData);
        new01_tr = net01.getTestResult();

        net02_s04.setFrozenWeights(true);
//        net02_s05.setFrozenWeights(true);

        System.out.println(
                " n1: " + new01_tr.getTestAccuracy()
                        + " n2: " + new02_tr.getTestAccuracy()
        );

        net03.train(trainData, epochs);
        net03.test(trainData);
        TestResult new03_tr = net03.getTestResult();

        net01.test(trainData);
        new01_tr = net01.getTestResult();

        net02.test(trainData);
        new02_tr = net02.getTestResult();

        net03_s06.setFrozenWeights(true);
//        net03_s07.setFrozenWeights(true);

        System.out.println(
                " n1: " + new01_tr.getTestAccuracy()
                        + " n2: " + new02_tr.getTestAccuracy()
                        + " n3: " + new03_tr.getTestAccuracy()
        );

        assertEquals(0.6037735849056604, new01_tr.getTestAccuracy(), 0.001);
        assertEquals(0.660377358490566, new02_tr.getTestAccuracy(), 0.001);
        assertEquals(0.6415094339622641, new03_tr.getTestAccuracy(), 0.001);
    }

}
