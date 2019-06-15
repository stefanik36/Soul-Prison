package com.stefanik36.soul_prison.source;

import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.stefanik36.soul_prison.data.DataResultTuple;
import io.vavr.collection.List;

import java.io.*;
import java.util.Optional;


public class BreastCancerWisconsinData {
    private static final String FILE_NAME = "Breast-Cancer-Wisconsin.csv";
    private static final String PATH = "src/main/resources/data/";
    private static final String JAR_PATH = "data/";
    private List<DataResultTuple> data;

    public BreastCancerWisconsinData() {
        this.data = List.empty();

        File file = getFile();
        try (CSVReader csvReader = new CSVReaderBuilder(
                new FileReader(file))
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

    public File extract(String filePath) {
        try {
            File f = File.createTempFile(filePath, null);
            FileOutputStream resourceOS = new FileOutputStream(f);
            byte[] byteArray = new byte[1024];
            int i;
            InputStream classIS = Optional.ofNullable(getClass().getClassLoader().getResourceAsStream(filePath)).orElseThrow(() -> new RuntimeException("cannot find file"));
            while ((i = classIS.read(byteArray)) > 0) {
                resourceOS.write(byteArray, 0, i);
            }
            classIS.close();
            resourceOS.close();
            return f;
        } catch (Exception e) {
            throw new RuntimeException("An error has occurred while extracting the database.");
        }
    }

    private File getFile() {
        File file = new File(PATH + FILE_NAME);
        if (file.exists()) {
            return file;
        }
        return extract(JAR_PATH + FILE_NAME);
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

