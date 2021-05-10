//
//  LineProgressBarView.swift
//  Pages
//
//  Created by 강지원 on 2020/11/15.
//

import UIKit

class LineProgressBarView: UIView {
    var progressView = UIProgressView()
    var progressLabel = UILabel()
    let progress = Progress(totalUnitCount: 100)
    let labelText = ""
    
    func updateSizeofView(labelText: String, frame: CGRect){
        self.frame = frame
        self.addSubview(progressView)
        self.addSubview(progressLabel)
        
        progressView.frame = CGRect(x: 0, y: self.frame.height/2, width: self.frame.width, height: self.frame.height/2)
        progressLabel.frame = CGRect(x: 0, y: 0, width: self.frame.width, height: self.frame.height/2)
        
        progressView.progress = 0.0
        progress.completedUnitCount = 0
        progressLabel.textAlignment = .left
        progressLabel.text = labelText
    }
    
    func updateProgressBarView(labelText: String, percent: Double, color: UIColor){
        Timer.scheduledTimer(withTimeInterval: 0.005, repeats: true) { (timer) in
            guard self.progress.isFinished == false else {
                timer.invalidate()
                return
            }
            
            self.progress.completedUnitCount += 1
            self.progressView.setProgress(Float(self.progress.fractionCompleted * percent/100), animated: true)
            self.progressView.progressTintColor = color
            self.progressView.transform.scaledBy(x: 1, y: 10)
            
            self.progressLabel.text = "\(labelText): \(percent)%"
            
        }
    }
    
}
