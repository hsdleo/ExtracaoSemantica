package app;

import org.opencv.core.Core;

public class App {

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		// new AppController().CreateThreadSensorColor();
		System.out.println("Comecei");
		int a = TakePicture();
		System.out.println("terminei");

	}
	public static int TakePicture() {
		CaptureData cp = new CaptureData();
		cp.TakePicture();
		//return cp.ProcessOCR();
		return 1;
	}

}
