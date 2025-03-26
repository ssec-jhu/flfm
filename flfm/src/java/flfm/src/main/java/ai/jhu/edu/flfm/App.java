package ai.jhu.edu.flfm;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;

import java.io.InputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.List;
import java.awt.image.BufferedImage;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

import org.apache.commons.imaging.ImageFormats;
import org.apache.commons.imaging.Imaging;





/**
 * Hello world!
 */
public class App {

    public static void openArray(String fName) throws Exception {
        NDManager manager = NDManager.newBaseManager();
        File tiffFile = new File(fName);

        // Extract individual pages as BufferedImages
        List<?> imagePages = Imaging.getAllBufferedImages(tiffFile);

        for (Object page : imagePages) {
            // Convert each page to NDArray
            Image image = ImageFactory.getInstance().fromImage((BufferedImage) page);
            NDArray ndArray = image.toNDArray(manager);

            // Do something with the NDArray
            System.out.println(ndArray.getShape());
        }
    }



    public static void main(String[] args) throws Exception, IOException, MalformedModelException, TranslateException  {
        System.out.println("Hello World!");
        Path imageFile = Paths.get("/home/ryan/repos/flfm/data/yale/light_field_image.tif");
        Path psfFile = Paths.get("/home/ryan/repos/flfm/data/yale/measured_psf.tif");

        openArray("/home/ryan/repos/flfm/data/yale/light_field_image.tif");

        // Open image using Input stream
        InputStream is = new FileInputStream(imageFile.toFile());
        Image img = ImageFactory.getInstance().fromInputStream(is);
        // Image img = ImageFactory.getInstance().fromFile(imageFile);
        System.out.println("Loaded image!");
        InputStream psfis = new FileInputStream(psfFile.toFile());
        Image psf = ImageFactory.getInstance().fromInputStream(psfis);
        // Image psf = ImageFactory.getInstance().fromFile(psfFile);
        System.out.println("Loaded psf!");

        try (Model model = Model.newInstance("flfm", "PyTorch")) {
            System.out.println("Loading model...");
            model.load(Paths.get("/home/ryan/repos/flfm/notebooks/richardson_lucy_10.pt"));
            System.out.println("Model loaded!");

            try (Predictor<Image[], Image[]> predictor = model.newPredictor(new MyTranslator())) {
                System.out.println("Predicting...");
                // time the call
                long start = System.currentTimeMillis();
                predictor.predict(new Image[]{img, psf});
                long end = System.currentTimeMillis();
                System.out.println("Prediction took " + (end - start) + " ms");
            }
        }
    }
}
