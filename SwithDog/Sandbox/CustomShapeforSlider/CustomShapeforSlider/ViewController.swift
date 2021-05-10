//
//  ViewController.swift
//  CustomShapeforSlider
//
//  Created by 강지원 on 2020/10/28.
//

import UIKit

class ViewController: UIViewController {

    var progressBarwithCode : CircularProgressBar! = CircularProgressBar(frame: CGRect(x: 200, y: 200, width: 200, height: 200))
    
    override func viewDidLoad() {
        super.viewDidLoad()
        progressBarwithCode.setMyEndAngle(CGFloat.pi)
        
        self.view.addSubview(progressBarwithCode)
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(self.handleTap(_:)))
        tapGesture.delegate = self
        self.view.addGestureRecognizer(tapGesture)
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        progressBarwithCode.setProgress(to: 1, withAnimation: true)
    }
}

extension ViewController: UIGestureRecognizerDelegate{
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
        return self.progressBarwithCode.frame.contains(touch.location(in: self.view))
    }
}

