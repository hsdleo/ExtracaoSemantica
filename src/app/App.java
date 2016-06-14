package app;

public class App {

	public static void main(String[] args) {
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
