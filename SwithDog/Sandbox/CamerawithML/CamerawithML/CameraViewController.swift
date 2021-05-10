//
//  CameraViewController.swift
//  CamerawithML
//
//  Created by 강지원 on 2020/10/26.
//

import UIKit
import AVFoundation
import CoreML
import Vision

class CameraViewController: UIViewController, AVCapturePhotoCaptureDelegate, AVCaptureVideoDataOutputSampleBufferDelegate{

    var exerciseHistory : [String] = ["standStill"]
    
    var captureSession = AVCaptureSession()
    var videoOutput = AVCaptureVideoDataOutput()
    
    private lazy var previewLayer: AVCaptureVideoPreviewLayer = {
        let preview = AVCaptureVideoPreviewLayer(session: self.captureSession)
        preview.videoGravity = .resizeAspect
        return preview
    }()
    
    @IBOutlet var exerciseRepsLabel: [UILabel]!
    
    @IBOutlet weak var dogImageView: UIImageView!
    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var showClassification: UILabel!
    
    @IBOutlet weak var startCameraButton: UIButton!
    @IBAction func startCamera(_ sender: UIButton) {
        startCameraButton.isSelected = true

        dogImageView.image = UIImage(named: "dog")
        
        // Take photo and assign it to photo output
        
    }
    
    @IBAction func stopCamera(_ sender: UIButton) {
        startCameraButton.isSelected = false
        self.captureSession.stopRunning()
        dogImageView.alpha = 0
    }
    
    private func addCameraInput() {
        guard let device = AVCaptureDevice.default(AVCaptureDevice.DeviceType.builtInWideAngleCamera, for: .video, position: .front) else {
            return}
        let cameraInput = try! AVCaptureDeviceInput(device: device)
        self.captureSession.addInput(cameraInput)
    }
    
    private func addPreviewLayer() {
        self.cameraView.layer.addSublayer(self.previewLayer)
    }
    
    private func addVideoOutput() {
        self.videoOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as NSString) : NSNumber(value: kCVPixelFormatType_32BGRA)] as [String : Any]
        self.videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "my.image.handling.queue"))
        self.captureSession.addOutput(self.videoOutput)
    }
    
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard CMSampleBufferGetImageBuffer(sampleBuffer) != nil else {
            debugPrint("unable to get image from sample buffer")
            return
        }
        print("did receive image frame")
        // process image here
        
        
        
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.previewLayer.frame = self.cameraView.bounds
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()

        self.addCameraInput()
        self.addPreviewLayer()
        self.addVideoOutput()
        self.captureSession.startRunning()
    }
    
    // ML Part
    
    lazy var classificationRequest: VNCoreMLRequest = {
    do {
        // Create model instance
       let model = try VNCoreMLModel(for: MyImageClassifier_1().model)
        
        // Request for model classification
       let request = VNCoreMLRequest(model: model, completionHandler: {   [weak self] request, error in
             self?.processClassifications(for: request, error: error)
    })
        // Image Processing choice
       request.imageCropAndScaleOption = .centerCrop
       return request
    } catch {
       fatalError("Failed to load Vision ML model: \(error)")
    }}()

    func createClassificationsRequest(for image: UIImage) {
        showClassification.text = "Classifying..."
        guard let ciImage = CIImage(image: image)
        else {
          fatalError("Unable to create \(CIImage.self) from \(image).")
        }
        
        // Process request in background
        DispatchQueue.global(qos: .userInitiated).async {
           let handler = VNImageRequestHandler(ciImage: ciImage)
           do {
            try handler.perform([self.classificationRequest])
           }catch {
            print("Failed to perform \n\(error.localizedDescription)")
           }
        }
    }
    
    func processClassifications(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
           guard let results = request.results
           else {
             self.showClassification.text = "Unable to classify image.\n\(error!.localizedDescription)"
             return
           }
            let classifications = results as! [VNClassificationObservation]
            if classifications.isEmpty {
                self.showClassification.text = "Nothing recognized."
            } else {
                let topClassifications = classifications.prefix(1)
                //let descriptions = topClassifications.map { classification in
                //return String(format: "(%.2f) %@", classification.confidence, classification.identifier)
                let descriptions = topClassifications.map { classification in
                    return String(format: "You are doing %@", classification.identifier)
                }
                self.showClassification.text = descriptions[0]
                
                // CountReps
                let identifierString = String(topClassifications[0].identifier).components(separatedBy: "-")[0]
                            
                if topClassifications[0].confidence > 0.97{
                    self.exerciseHistory.append(identifierString)
                    if self.exerciseHistory[self.exerciseHistory.count-2] == "StandStill" && self.exerciseHistory[self.exerciseHistory.count-2] != identifierString{
                        exerciseList[identifierString]?.numberOfReps += 1
                    }
                }
                
                // Update Exercise OutletCollection
                self.exerciseRepsLabel[0].text = "Squat: \(String(describing: exerciseList["Squat"]?.numberOfReps))"
                self.exerciseRepsLabel[1].text = "Lunge: \(String(describing: exerciseList["Lunge"]?.numberOfReps))"
                self.exerciseRepsLabel[2].text = "J.J: \(String(describing: exerciseList["JumpingJack"]?.numberOfReps))"
            }
        }
    }
    
    // Other Implementations
    func resetReps(){
        squat.numberOfReps = 0
        lunge.numberOfReps = 0
        jumpingJack.numberOfReps = 0
        standStill.numberOfReps = 0
    }
 
    
}
