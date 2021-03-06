package com.stefanik36.soul_prison.source;

import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.stefanik36.soul_prison.data.DataResultTuple;
import io.vavr.collection.List;

import java.io.FileReader;
import java.io.IOException;

public class RedWineQualityData {

    private List<DataResultTuple> data;

    public RedWineQualityData() {
        this.data = List.empty();
        try (CSVReader csvReader = new CSVReaderBuilder(
                new FileReader("src/main/resources/data/wine-quality-red.csv"))
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

    public List<DataResultTuple> getData() {
        return data;
    }
}

