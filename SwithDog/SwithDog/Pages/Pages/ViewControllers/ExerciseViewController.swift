//
//  ExerciseViewController.swift
//  Pages
//
//  Created by 강지원 on 2020/11/13.
//

import UIKit
import AVFoundation
import CoreML
import Vision
import VideoToolbox
import QuartzCore

class ExerciseViewController: UIViewController,AVCapturePhotoCaptureDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // MARK: - Variables
    var cameraButtonpushNumber = 0
    
    var fiveList : [String] = []
    var tenList : [String] = []
    let today = NSDate()
    let formatter = DateFormatter()
    
    var time = 5
    var intTime = 0
    var timer = Timer()
    var startTimer = false
    
    var randomExerciseList : [String] = []
    var exerciseHistory : [String] = []
    
    var captureSession = AVCaptureSession()
    var videoOutput = AVCaptureVideoDataOutput()
    
    var audioPlayer = AVAudioPlayer()
    
    private lazy var previewLayer: AVCaptureVideoPreviewLayer = {
        let preview = AVCaptureVideoPreviewLayer(session: self.captureSession)
        preview.videoGravity = .resizeAspect
        return preview
    }()
    
    // MARK: - Background
    let backgroundPattern = UIImageView()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .secondarySystemBackground
        
        backgroundPattern.image = UIImage(named: "backgroundPattern(gray)")
        backgroundPattern.contentMode = .scaleAspectFill
        backgroundPattern.clipsToBounds = true
        self.view.addSubview(backgroundPattern)
        
        backgroundPattern.translatesAutoresizingMaskIntoConstraints = false
        backgroundPattern.topAnchor.constraint(equalTo: view.topAnchor).isActive = true
        backgroundPattern.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 1).isActive = true
        backgroundPattern.bottomAnchor.constraint(equalTo: view.bottomAnchor).isActive = true
        
        
        // MARK: - ViewDidLoad
        cameraButtonpushNumber = 0
        initAutoLayout()
        additionalFetch()
        
        self.addCameraInput()
        self.addPreviewLayer()
        self.addVideoOutput()
        
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.previewLayer.frame = self.cameraView.bounds
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(true)
        dogImageView.image = dogImageList[nowUser.currentPlan]
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(true)
        
        let today = Date()
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
        let stringToday = dateFormatter.string(from: today)
        
        let plan = planList[nowUser.currentPlan]
        let todayReps = (nowUser.pastSquatRepsDict[stringToday] ?? 0) + (nowUser.pastLungeRepsDict[stringToday] ?? 0) + (nowUser.pastJumpingJackRepsDict[stringToday] ?? 0)
        let designatedReps = plan.squatRepsPerDay + plan.lungeRepsPerDay + plan.jumpingJackRepsPerDay
        let todayAchievement = (todayReps)/(designatedReps)

        if nowUser.currentPlan == 0{
        }else{
            nowUser.currentPlanAchievementRate = nowUser.currentPlanAchievementRate + Double((todayAchievement) / (planList[nowUser.currentPlan].numberOfDays))
        }
    }
    
    // MARK: - Elements
    var explainLabel1 = UILabel()
    var exerciseRepsLabel = UILabel()
    var whiteView = UIView()
    var dogImageView = UIImageView()
    var cameraView = UIView()
    var exerciseGameLabel = UILabel()
    var controlCameraButton = UIButton()
    
    
    // MARK: - AutoLayout
    func initAutoLayout(){
        self.view.addSubview(explainLabel1)
        self.view.addSubview(exerciseRepsLabel)
        self.view.addSubview(whiteView)
        self.view.addSubview(dogImageView)
        self.view.addSubview(cameraView)
        self.view.addSubview(exerciseGameLabel)
        self.view.addSubview(controlCameraButton)

        explainLabel1.translatesAutoresizingMaskIntoConstraints = false
        exerciseRepsLabel.translatesAutoresizingMaskIntoConstraints = false
        whiteView.translatesAutoresizingMaskIntoConstraints = false
        dogImageView.translatesAutoresizingMaskIntoConstraints = false
        cameraView.translatesAutoresizingMaskIntoConstraints = false
        exerciseGameLabel.translatesAutoresizingMaskIntoConstraints = false
        controlCameraButton.translatesAutoresizingMaskIntoConstraints = false

        // MARK: - explainLabel1
        explainLabel1.leftAnchor.constraint(equalTo: exerciseRepsLabel.leftAnchor, constant: 7).isActive = true
        explainLabel1.bottomAnchor.constraint(equalTo: exerciseRepsLabel.topAnchor, constant: -5).isActive = true
        
        explainLabel1.text = "오늘 운동 횟수"
        explainLabel1.backgroundColor = .clear
        explainLabel1.font = UIFont.boldSystemFont(ofSize: 20)
        explainLabel1.textAlignment = .left
        
        // MARK: - exerciseRepsLabel
        exerciseRepsLabel.centerXAnchor.constraint(equalTo: self.view.centerXAnchor).isActive = true
        exerciseRepsLabel.widthAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.widthAnchor, constant: -40).isActive = true
        exerciseRepsLabel.topAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.topAnchor, constant: 60).isActive = true
        exerciseRepsLabel.heightAnchor.constraint(equalToConstant: 60).isActive = true
        
        exerciseRepsLabel.layer.cornerRadius = 15
        exerciseRepsLabel.clipsToBounds = true
        
        exerciseRepsLabel.backgroundColor = .white
        exerciseRepsLabel.text = "스쿼트: 0회,   런지: 0회,   팔벌려뛰기: 0회"
        exerciseRepsLabel.numberOfLines = 0
        exerciseRepsLabel.textAlignment = .center
        exerciseRepsLabel.adjustsFontSizeToFitWidth = true
        
        // MARK: - whiteViewUnderDogImage
        whiteView.centerXAnchor.constraint(equalTo: self.view.centerXAnchor).isActive = true
        whiteView.widthAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.widthAnchor, constant: -40).isActive = true
        whiteView.topAnchor.constraint(equalTo: self.exerciseRepsLabel.bottomAnchor, constant: 20).isActive = true
        whiteView.bottomAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.bottomAnchor, constant: -20).isActive = true
        
        whiteView.backgroundColor = .white
        whiteView.layer.cornerRadius = 20
        whiteView.clipsToBounds = true
        
        // MARK: - dogImageView
        dogImageView.centerXAnchor.constraint(equalTo: self.view.centerXAnchor).isActive = true
        dogImageView.widthAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.widthAnchor, constant: -40).isActive = true
        dogImageView.topAnchor.constraint(equalTo: self.exerciseRepsLabel.bottomAnchor, constant: 20).isActive = true
        dogImageView.bottomAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.bottomAnchor, constant: -20).isActive = true
        
        dogImageView.contentMode = .scaleAspectFill
        dogImageView.layer.cornerRadius = 20
        dogImageView.clipsToBounds = true
        
        dogImageView.image = dogImageList[nowUser.currentPlan]
        
        // MARK: - cameraView
        cameraView.centerXAnchor.constraint(equalTo: self.view.centerXAnchor).isActive = true
        cameraView.widthAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.widthAnchor, constant: -40).isActive = true
        cameraView.topAnchor.constraint(equalTo: exerciseRepsLabel.bottomAnchor, constant: 20).isActive = true
        cameraView.bottomAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.bottomAnchor, constant: -20).isActive = true
        
        cameraView.layer.cornerRadius = 20
        cameraView.clipsToBounds = true
        
        // MARK: - exerciseGameLabel
        exerciseGameLabel.centerXAnchor.constraint(equalTo: self.view.centerXAnchor).isActive = true
        exerciseGameLabel.widthAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.widthAnchor, constant: -40).isActive = true
        exerciseGameLabel.topAnchor.constraint(equalTo: cameraView.topAnchor).isActive = true
        exerciseGameLabel.heightAnchor.constraint(equalToConstant: 20).isActive = true
        
        exerciseGameLabel.layer.cornerRadius = 5
        exerciseGameLabel.clipsToBounds = true
        
        exerciseGameLabel.text = ""
        exerciseGameLabel.backgroundColor = .clear
        exerciseGameLabel.textColor = .black
        exerciseGameLabel.numberOfLines = 0
        exerciseGameLabel.textAlignment = .center
        exerciseGameLabel.adjustsFontSizeToFitWidth = true
        
        // MARK: - controlCameraButton
        controlCameraButton.bottomAnchor.constraint(equalTo: cameraView.bottomAnchor, constant: -10).isActive = true
        controlCameraButton.trailingAnchor.constraint(equalTo: cameraView.trailingAnchor, constant: -10).isActive = true
        controlCameraButton.widthAnchor.constraint(equalToConstant: 80).isActive = true
        controlCameraButton.heightAnchor.constraint(equalToConstant: 40).isActive = true
        
        controlCameraButton.layer.cornerRadius = 15
        controlCameraButton.clipsToBounds = true

        controlCameraButton.setTitle("START", for: .normal)
        controlCameraButton.setTitleColor(.white, for: .normal)
        controlCameraButton.backgroundColor = #colorLiteral(red: 0, green: 0.7016062737, blue: 0.6565710306, alpha: 1)
        controlCameraButton.addTarget(self, action: #selector(pushCameraButton), for: .touchUpInside)
        
    }
    
    // MARK: - Objc Functions
    @objc func updateExerciseGameLabel(){
        if time <= (randomExerciseList.count*5){
            intTime = time/5 - 1
            time += 5
            
            let nowExercise : String = randomExerciseList[intTime]
            exerciseGameLabel.text = "지금 할 운동은 \(nowExercise)!!!"
            soundPlayGame(filename: nowExercise)
            
        }else{
            stopTime()
        }
    }
    
    @objc func pushCameraButton(){
        if cameraButtonpushNumber%2 == 0{
            dogImageView.alpha = 0
            whiteView.alpha = 0
                        
            self.exerciseHistory = Array(repeating: "StandStill", count: 30)
            self.randomExerciseList = []
            self.tenList = []
            self.fiveList = []
                        
            prepareExerciseGame(squatNum: planList[nowUser.currentPlan].squatRepsPerDay, lungeNum: planList[nowUser.currentPlan].lungeRepsPerDay, jumpingJackNum: planList[nowUser.currentPlan].jumpingJackRepsPerDay)
            
            self.captureSession.startRunning()
            
            controlCameraButton.setTitle("STOP", for: .normal)
            controlCameraButton.setTitleColor(.white, for: .normal)
            cameraButtonpushNumber += 1
            
            soundStartGame()
            exerciseGameLabel.text = "지금 할 운동은 \(randomExerciseList[0])!!!"

            startTime()
            
        }else{
            stopTime()
            audioPlayer.stop()
            self.captureSession.stopRunning()
            dogImageView.alpha = 1
            whiteView.alpha = 1
            cameraView.alpha = 0
            
            exerciseGameLabel.text = "운동 언제 하실거에요~?"
            exerciseGameLabel.backgroundColor = .clear
            
            controlCameraButton.setTitle("Start", for: .normal)
            cameraButtonpushNumber += 1
        }
    }
    
    // MARK: - Exercise Game
    func startTime(){
        if startTimer == false{
            startTimer = true
            timer = Timer.scheduledTimer(timeInterval: 5, target: self, selector: #selector(self.updateExerciseGameLabel), userInfo: nil, repeats: true)
        }
    }
    
    func stopTime(){
        startTimer = false
        timer.invalidate()
    }
    
    func prepareExerciseGame(squatNum: Int, lungeNum: Int, jumpingJackNum: Int){
        for _ in 1...squatNum{
            randomExerciseList.append("Squat")
        }
        
        for _ in 1...lungeNum{
            randomExerciseList.append("Lunge")
        }
        
        for _ in 1...jumpingJackNum{
            randomExerciseList.append("JumpingJack")
        }
        
        randomExerciseList.shuffle()
        //startTime()
    }
    
    func updateExerciseRepsLabel(){
        formatter.dateFormat = "yyyy-MM-dd"
        let stringDate = formatter.string(from: today as Date)
        
        exerciseRepsLabel.text = "\(stringDate) \n 스쿼트: \(nowUser.pastSquatRepsDict[stringDate] ?? 0) 회,   Lunge: \(nowUser.pastLungeRepsDict[stringDate] ?? 0) 회,   팔벌려뛰기: \(nowUser.pastJumpingJackRepsDict[stringDate] ?? 0) 회"
    }
    
    func countWordNuminArray(list: Array<String>, word: String) -> Int{
        var num = 0
        for element in list{
            if element == word{
                num += 1
            }
        }
        return num
    }
    
    // MARK: - Audio Part
    func soundStartGame() {
         guard let sound = NSDataAsset(name: "soonPlay") else {print("Asset load error"); return}
         do {
             try self.audioPlayer = AVAudioPlayer(data: sound.data)
             self.audioPlayer.prepareToPlay()
         } catch let error as NSError {
             print("\(error.localizedDescription)")
         }
        audioPlayer.play()
     }
    
    func soundPlayGame(filename: String){
        guard let sound = NSDataAsset(name: filename) else {print("Asset load error"); return}
        do {
            try self.audioPlayer = AVAudioPlayer(data: sound.data)
            self.audioPlayer.prepareToPlay()
        } catch let error as NSError {
            print("\(error.localizedDescription)")
        }
       audioPlayer.play()
    }
    
    // MARK: - Camera Part
    private func addCameraInput() {
        guard let device = AVCaptureDevice.default(AVCaptureDevice.DeviceType.builtInWideAngleCamera, for: .video, position: .front) else {fatalError("no camera found") }
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
        guard let frame = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            debugPrint("unable to get image from sample buffer")
            return
        }
        print("did receive image frame")
        // process image here
        
        var cgImage: CGImage!
        VTCreateCGImageFromCVPixelBuffer(frame, options: nil, imageOut: &cgImage)
        
        createClassificationsRequest(for: UIImage(cgImage: cgImage))
    }
    
    // MARK: - ML Part
    lazy var classificationRequest: VNCoreMLRequest = {
    do {
        // Create model instance
       let model = try VNCoreMLModel(for: Woongbi().model)

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
        print("Classifying...")
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
             print("Unable to classify image.\n\(error!.localizedDescription)")
             return
           }
            let classifications = results as! [VNClassificationObservation]
            if classifications.isEmpty {
                print("Nothing recognized.")
            } else {

                //MARK: - Post Processing
                let topClassifications = classifications.prefix(1)
                
                let topIdentifier: String = topClassifications[0].identifier
                let topConfidence: Float = topClassifications[0].confidence
                
                print("\(topIdentifier) ,\(topConfidence)")

                // MARK: - CountReps
                if topConfidence > 0.95{
                    self.exerciseHistory.append(topIdentifier)
                }else{
                    self.exerciseHistory.append("StandStill")
                }
                
                self.fiveList = []
                self.tenList = []
                
                for i in 17...21{
                    self.fiveList.append(self.exerciseHistory[self.exerciseHistory.count - i])
                }

                for i in 2...16{
                    self.tenList.append(self.exerciseHistory[self.exerciseHistory.count - i])
                }
                
                let countOfTopIdentifier = self.countWordNuminArray(list: self.tenList, word: topIdentifier)
                
                let countOfStandStill = self.countWordNuminArray(list: self.fiveList, word: "StandStill")
                
                if countOfStandStill > 4 && countOfTopIdentifier > 12{
                    self.formatter.dateFormat = "yyyy-MM-dd"
                    let stringDate = self.formatter.string(from: self.today as Date)
                    if topIdentifier == "Squat"{
                        if nowUser.pastSquatRepsDict[stringDate] == nil{
                            nowUser.pastSquatRepsDict[stringDate] = 1
                        }else{
                            nowUser.pastSquatRepsDict[stringDate]! += 1
                        }
                    }else if topIdentifier == "Lunge"{
                        if nowUser.pastLungeRepsDict[stringDate] == nil{
                            nowUser.pastLungeRepsDict[stringDate] = 1
                        }else{
                            nowUser.pastLungeRepsDict[stringDate]! += 1
                        }
                    }else if topIdentifier == "JumpingJack"{
                        if nowUser.pastJumpingJackRepsDict[stringDate] == nil{
                            nowUser.pastJumpingJackRepsDict[stringDate] = 1
                        }else{
                            nowUser.pastJumpingJackRepsDict[stringDate]! += 1
                        }
                    }

                    for _ in 1...30{
                        self.exerciseHistory.append("StandStill")
                    }
                    self.updateExerciseRepsLabel()
                }
            }
        }
    }
}

