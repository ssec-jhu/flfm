package ai.jhu.edu.flfm;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;

import javax.annotation.processing.SupportedOptions;

import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.io.Opener;
import ij.process.ImageProcessor;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

@Command(name = "ExampleCLI", description = "An example CLI app with four arguments using Picocli.")
public class App implements Runnable {

    @Option(names = {"-i", "--image"}, description = "Path to the input image", required = true)
    private String imageString;
    @Option(names = {"-p", "--psf"}, description = "Path to the PSF image", required = true)
    private String psfString;
    @Option(names = {"-m", "--model"}, description = "Path to the model file", required = true)
    private String modelString;
    @Option(names = {"-o", "--output"}, description = "Path to the output image", required = true)
    private String outputString;


    /**
     * Process the image using
     * @param imagePath The path to the input image.
     * @param psfPath The path to the PSF image.
     * @param modelPath The path to the model file.
     * @param outputPath The path to save the output image.
     * @throws Exception
     * @throws IOException
     * @throws MalformedModelException
     * @throws TranslateException
     */
    public static void process(String imagePath, String psfPath, String modelPath, String outputPath) throws Exception, IOException, MalformedModelException, TranslateException {

        try(NDManager manager = NDManager.newBaseManager()){
            NDArray image = ArrayIO.loadArray(imagePath, manager);
            NDArray psf = ArrayIO.loadArray(psfPath, manager);
            NDArray normedPsf = psf.div(psf.sum());

            try (Model model = Model.newInstance("flfm", "PyTorch")) {
                model.load(Paths.get(modelPath));

                try (Predictor<NDArray[], NDArray> predictor = model.newPredictor(new MyTranslator())) {
                    long start = System.currentTimeMillis();
                    NDArray out = predictor.predict(new NDArray[]{image, normedPsf});
                    long end = System.currentTimeMillis();
                    System.out.println("Prediction took " + (end - start) + " ms");
                    ArrayIO.saveArray(outputPath, out);
                }
            }
        }
    }

    @Override
    public void run() {
        try {
            process(imageString, psfString, modelString, outputString);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws Exception, IOException, MalformedModelException, TranslateException  {
        CommandLine.run(new App(), args);
    }
}
