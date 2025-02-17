package ssec_jhu;

import org.tensorflow.Tensor;
import org.tensorflow.Result;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.proto.ConfigProto;
import org.tensorflow.proto.GPUOptions;
import org.tensorflow.types.TFloat32;

import java.awt.image.BufferedImage;
import java.io.IOError;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.time.Duration;
import java.time.Instant;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

public class App {
  private static FloatNdArray loadImageAsArray(String path) throws IOException {

    System.out.println("Loading image from: " + path);

    // Read the image file
    ImageInputStream input = ImageIO.createImageInputStream(new java.io.File(path));
    Iterator<ImageReader> readers = ImageIO.getImageReaders(input);

    if (!readers.hasNext()) {
      throw new IOException("No ImageReaders found for given file format.");
    }

    ImageReader reader = readers.next();
    reader.setInput(input);

    int numImages = reader.getNumImages(true);
    System.out.println("Number of images: " + numImages);

    BufferedImage[] images = new BufferedImage[numImages];
    for (int i = 0; i < numImages; i++) {
      images[i] = reader.read(i);
    }

    // Close the reader
    reader.dispose();

    // convert the image stack to a tensor
    int n = numImages;
    int h = images[0].getHeight();
    int w = images[0].getWidth();
    float[][][] imageData = new float[n][h][w];

    for (int i = 0; i < n; i++) {
      BufferedImage img = images[i];
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          short value = (short) (img.getRGB(x, y) & 0xFFFF);
          imageData[i][y][x] = value / 65535f;
        }
      }
    }

    FloatNdArray imageArray = StdArrays.ndCopyOf(imageData);
    System.out.println("Image loaded and converted to array. Shape: " + imageArray.shape());
    return imageArray;
  }

  public static void main(String[] args) {
    System.out.println("CWD:" + System.getProperty("user.dir"));
    String modelLocation = "./exported_model";

    // Open a TIFF image
    String imagePath = "./data/yale/light_field_image.tif";
    String psfPath = "./data/yale/measured_psf.tif";

    FloatNdArray image, psf, data;
    try {
      image = loadImageAsArray(imagePath);
      psf = loadImageAsArray(psfPath);
    } catch (IOException e) {
      System.out.println("Error: " + e.getMessage());
      throw new IOError(e);
    }

    data = NdArrays.ofFloats(psf.shape());
    // fill data with 0.5f
    for (int i = 0; i < data.shape().get(0); i++) {
      for (int j = 0; j < data.shape().get(1); j++) {
        for (int k = 0; k < data.shape().get(2); k++) {
          data.setFloat(0.5f, i, j, k);
        }
      }
    }

    GPUOptions gpu =
        GPUOptions.newBuilder()
            .setVisibleDeviceList("0")
            .setPerProcessGpuMemoryFraction(0.8)
            .setAllowGrowth(true)
            .build();

    ConfigProto configProto =
        ConfigProto.newBuilder()
            .setAllowSoftPlacement(true)
            .setLogDevicePlacement(true)
            .mergeGpuOptions(gpu)
            .build();

    SavedModelBundle model =
        SavedModelBundle.loader(modelLocation).withConfigProto(configProto).load();

    try (Tensor imageTensor = TFloat32.tensorOf(image);
        Tensor psfTensor = TFloat32.tensorOf(psf);
        Tensor dataTensor = TFloat32.tensorOf(data)) {

      Map<String, Tensor> inputs = new HashMap<String, Tensor>();
      inputs.put("data", dataTensor);
      inputs.put("image", imageTensor);
      inputs.put("psf", psfTensor);

      System.out.println("Running model...");
      Instant start = Instant.now();
      Result result = model.function("serving_default").call(inputs);
      Tensor output = result.get("output_0").get();
      Instant end = Instant.now();
      System.out.println(
          "Model run in " + (Duration.between(start, end).toMillis() / 1000f) + "seconds");
    }
    System.out.println("Done.");
  }
}
