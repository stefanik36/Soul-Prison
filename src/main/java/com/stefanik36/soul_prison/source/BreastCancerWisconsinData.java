package com.stefanik36.soul_prison.source;

import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.stefanik36.soul_prison.data.DataResultTuple;
import io.vavr.collection.List;

import java.io.FileReader;
import java.io.IOException;


public class BreastCancerWisconsinData {

    private List<DataResultTuple> data;

    public BreastCancerWisconsinData() {
        this.data = List.empty();
        try (CSVReader csvReader = new CSVReaderBuilder(
                new FileReader("src/main/resources/data/Breast-Cancer-Wisconsin.csv"))
                .withCSVParser(new CSVParserBuilder().withSeparator(';').build())
                .build()
        ) {

            String[] values;
            boolean first = true;
            while ((values = csvReader.readNext()) != null) {
                if (first) {
                    first = false;
                    continue;
                }
                data = data.append(
                        new DataResultTuple(
                                Double.valueOf(values[values.length - 1]),
                                List.of(values).dropRight(1).map(Double::valueOf)
                        )
                );
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    /**
     * ? to 0 changed
     *
     * @return BreastCancerWisconsinData
     */
    public List<DataResultTuple> getData() {
        return data;
    }
}

