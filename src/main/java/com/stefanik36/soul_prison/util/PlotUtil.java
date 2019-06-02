package com.stefanik36.soul_prison.util;

import com.stefanik36.soul_prison.model.TestResult;
import com.stefanik36.soul_prison.model.TrainSummary;
import io.vavr.collection.List;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.category.DefaultCategoryDataset;

import java.awt.*;

public class PlotUtil {

    public static void plotResult(List<TrainSummary> trainSummaries) {
        plotResult(trainSummaries, null);
    }

    public static void plotResult(List<TrainSummary> trainSummaries, TestResult tr) {
        DefaultCategoryDataset objDataset = new DefaultCategoryDataset();
        int sleep = 1000;
        int samples = 20;
        int samplingValue = trainSummaries.size() / samples;
        for (int i = 0; i < trainSummaries.size(); i++) {
            if (i % samplingValue == 0) {
                TrainSummary ts = trainSummaries.get(i);
                String sep = String.valueOf(i);
                objDataset.setValue(ts.getTrainAccuracy(), "TrainAccuracy", sep);
                objDataset.setValue(ts.getTrainLoss(), "TrainLoss", sep);
                objDataset.setValue(ts.getValidationAccuracy(), "ValidationAccuracy", sep);
                objDataset.setValue(ts.getValidationLoss(), "ValidationLoss", sep);
                if (tr != null) {
                    objDataset.setValue(tr.getTestAccuracy(), "TestAccuracy", sep);
                    objDataset.setValue(tr.getTestLoss(), "TestLoss", sep);
                }
            }
        }

        JFreeChart objChart = ChartFactory.createLineChart(
                "Train result",
                "epochs",
                "value",
                objDataset
        );

        CategoryPlot plot = objChart.getCategoryPlot();
        // here we change the line size
        int seriesCount = plot.getDataset().getRowCount();
        for (int i = 0; i < seriesCount; i++) {
            plot.getRenderer().setSeriesStroke(i, new BasicStroke(4));
        }
        ChartFrame frame = new ChartFrame("Demo", objChart);
        frame.pack();
        frame.setVisible(true);
        try {
            Thread.sleep(sleep * 1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
