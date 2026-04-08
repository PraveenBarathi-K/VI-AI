package com.surendramaran.yolov8tflite

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener, DepthEstimator.DepthListener, SensorEventListener, TextToSpeech.OnInitListener {
    private lateinit var binding: ActivityMainBinding
    private var detector: Detector? = null
    private var depthEstimator: DepthEstimator? = null
    private var tts: TextToSpeech? = null

    // Sensors and Fusion
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private val metricScaler = MetricScaler()
    private var currentMetricScale = 1.5f

    private var latestDetections: List<BoundingBox> = emptyList()
    private var frameCounter = 0
    private lateinit var cameraExecutor: ExecutorService
    private var lastUtteranceTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        tts = TextToSpeech(this, this)

        // Setup Sensors
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

        // Initialize Models then Camera
        cameraExecutor.execute {
            try {
                detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
                depthEstimator = DepthEstimator(baseContext, "midas_model.tflite", this)
                runOnUiThread {
                    if (allPermissionsGranted()) startCamera()
                    else requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Init Error: ${e.message}")
            }
        }
        bindListeners()
    }

    private fun bindListeners() {
        binding.isGpu.setOnCheckedChangeListener { _, isChecked ->
            cameraExecutor.submit { detector?.restart(isGpu = isChecked) }
            val color = if (isChecked) R.color.orange else R.color.gray
            binding.isGpu.setBackgroundColor(ContextCompat.getColor(baseContext, color))
        }
    }

    // --- Sensor Logic ---
    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type == Sensor.TYPE_LINEAR_ACCELERATION) {
            currentMetricScale = metricScaler.getStableScale(event.values, event.timestamp)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // --- Hybrid Depth Logic ---
    override fun onDepthMapGenerated(depthMap: Array<FloatArray>, inferenceTime: Long) {
        if (isFinishing || isDestroyed) return
        val size = depthMap.size
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)

        var maxDepthVal = 0f
        var hazardX = 0
        var hazardY = 0

        // Find min/max for dynamic heatmap normalization
        var mapMin = Float.MAX_VALUE
        var mapMax = Float.MIN_VALUE
        for (row in depthMap) {
            for (v in row) {
                if (v < mapMin) mapMin = v
                if (v > mapMax) mapMax = v
            }
        }
        val range = mapMax - mapMin

        for (y in 0 until size) {
            for (x in 0 until size) {
                val rawVal = depthMap[y][x]

                // Scan walking path (Y-axis 30% to 85%)
                if (y in (size * 0.3).toInt()..(size * 0.85).toInt()) {
                    if (rawVal > maxDepthVal) {
                        maxDepthVal = rawVal
                        hazardX = x
                        hazardY = y
                    }
                }

                val norm = if (range > 0f) (((rawVal - mapMin) / range) * 255).toInt().coerceIn(0, 255) else 0
                bitmap.setPixel(x, y, Color.rgb(norm, 0, 255 - norm))
            }
        }

        // Distance Calculation: Uses Inverse Scale to keep it realistic
        val distance = (1.05f - ( (maxDepthVal - mapMin) / (range + 0.001f) )) * currentMetricScale * 3.0f

        // Semantic Check: Is this depth hazard inside a YOLO detection?
        var objectLabel = "Obstacle"
        for (box in latestDetections) {
            val bx = (box.x1 * size).toInt()
            val by = (box.y1 * size).toInt()
            val bw = (box.w * size).toInt()
            val bh = (box.h * size).toInt()
            if (hazardX in bx..(bx + bw) && hazardY in by..(by + bh)) {
                objectLabel = box.clsName
                break
            }
        }

        val side = when {
            hazardX < size / 3 -> "Left"
            hazardX < 2 * size / 3 -> "Center"
            else -> "Right"
        }

        runOnUiThread {
            binding.depthOverlay.setImageBitmap(bitmap)

            // Alert Threshold: If normalized value is high (closer)
            val normalizedHazard = if (range > 0f) (maxDepthVal - mapMin) / range else 0f

            if (normalizedHazard > 0.75f) {
                val message = "$objectLabel $side, ${String.format("%.1f", distance.coerceAtLeast(0.5f))} meters"
                binding.inferenceTime.text = "STOP: $message"
                binding.inferenceTime.setTextColor(Color.RED)

                if (System.currentTimeMillis() - lastUtteranceTime > 3500) {
                    tts?.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
                    lastUtteranceTime = System.currentTimeMillis()
                }
            } else {
                binding.inferenceTime.text = "PATH CLEAR"
                binding.inferenceTime.setTextColor(Color.GREEN)
            }
        }
    }

    // --- Camera Pipeline ---
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build()
            val rotation = binding.viewFinder.display.rotation

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalyzer.setAnalyzer(cameraExecutor) { image ->
                val d = detector
                val de = depthEstimator
                if (d == null || de == null) {
                    image.close()
                    return@setAnalyzer
                }

                val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                image.use { bitmap.copyPixelsFromBuffer(image.planes[0].buffer) }

                val matrix = Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }
                val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

                d.detect(rotated)
                if (frameCounter % 12 == 0) de.estimate(rotated)
                frameCounter++
                image.close()
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer)
            preview.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onInit(status: Int) { if (status == TextToSpeech.SUCCESS) tts?.language = Locale.US }
    override fun onEmptyDetect() { latestDetections = emptyList() ; runOnUiThread { binding.overlay.clear() } }
    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        latestDetections = boundingBoxes
        runOnUiThread {
            binding.overlay.setResults(boundingBoxes)
            binding.overlay.invalidate()
        }
    }

    override fun onResume() {
        super.onResume()
        accelerometer?.also { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
        if (allPermissionsGranted() && detector != null) startCamera()
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        depthEstimator?.close()
        tts?.shutdown()
        cameraExecutor.shutdown()
    }



    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { if (it[Manifest.permission.CAMERA] == true) startCamera() }
    companion object { private const val TAG = "Camera" ; private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA) }
}