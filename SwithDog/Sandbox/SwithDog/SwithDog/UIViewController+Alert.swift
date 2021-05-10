//
//  UIViewController+Alert.swift
//  SwithDog
//
//  Created by Leekyujin on 2020/10/21.
//  Copyright © 2020 Leekyujin. All rights reserved.
//

import UIKit



extension UIViewController {
    func ratingSheet(title: String = "평가하기", message: String) {
        let sheet = UIAlertController(title: title, message: message, preferredStyle: .actionSheet)
        
        let okAction = UIAlertAction(title: "보내기", style: .default, handler: nil)
        let cancleAction = UIAlertAction(title: "나중에 하기", style: .destructive, handler: nil)
        sheet.addAction(okAction)
        sheet.addAction(cancleAction)
        
        
        
        present(sheet, animated: true, completion: nil)
    }
}



extension UIViewController {
    func alert(title: String = "알림", message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        
        let okAction = UIAlertAction(title: "확인", style: .default) {(copy) in
            UIPasteboard.general.string = "sto-leege@naver.com"
        }
        let cancelAction = UIAlertAction(title: "취소", style: .destructive, handler: nil)
        alert.addAction(okAction)
        alert.addAction(cancelAction)
        
        present(alert, animated: true, completion: nil)
    }
    }


extension UIViewController {
func urlAlert(title: String = "사파리에서 열기", message: String) {
    let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
    
    let okAction = UIAlertAction(title: "확인", style: .default) {(action) in
        UIApplication.shared.open(NSURL(string: "http://www.animal.go.kr")! as URL)
    }
    let cancelAction = UIAlertAction(title: "취소", style: .destructive, handler: nil)
    alert.addAction(okAction)
    alert.addAction(cancelAction)
    
    present(alert, animated: true, completion: nil)
}

}
