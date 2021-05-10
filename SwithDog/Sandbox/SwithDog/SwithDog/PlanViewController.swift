//
//  PlanViewController.swift
//  SwithDog
//
//  Created by Leekyujin on 2020/10/21.
//  Copyright © 2020 Leekyujin. All rights reserved.
//

import UIKit

class PlanViewController: UIViewController {
    
    
    //데이트피커 ----------------------------------------------
    @IBOutlet weak var textField: UITextField!
    
    var datePicker: UIDatePicker!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //CoreData ---------------------------------------------
        (UIApplication.shared.delegate as! AppDelegate).persistentContainer.viewContext

        
        //데이트피커 ----------------------------------------------
        setupdatePicker()
        // 알람 -------

    }

    func setupdatePicker() {
        self.datePicker = UIDatePicker.init(frame: CGRect(x: 0, y: 0, width: self.view.bounds.width, height: 200))
        datePicker.datePickerMode = .time
        datePicker.addTarget(self, action: #selector(self.dateChanged), for: .allEvents)
        
        
        if #available(iOS 13.4, *) {
            datePicker.preferredDatePickerStyle = .wheels
        }
        
        
        self.textField.inputView = datePicker
        
        let toolBar:UIToolbar = UIToolbar.init(frame: CGRect(x: 0, y: 0, width: self.view.bounds.width, height: 44))
        
        let spaceButton:UIBarButtonItem = UIBarButtonItem(barButtonSystemItem: UIBarButtonItem.SystemItem.flexibleSpace, target: nil, action: #selector(self.tapOnDoneBut))
        
        let doneButton:UIBarButtonItem = UIBarButtonItem(barButtonSystemItem: UIBarButtonItem.SystemItem.done, target: self, action: #selector(self.tapOnDoneBut))
        
        toolBar.setItems([spaceButton, doneButton], animated: true)
        
        self.textField.inputAccessoryView = toolBar
        
    }
    @objc func dateChanged(){
        
        let dateFormat = DateFormatter()
        dateFormat.dateStyle = .medium
        dateFormat.timeStyle = .short
        dateFormat.locale = Locale(identifier: "ko_KR")


        self.textField.text = dateFormat.string(from: datePicker.date)
        
        
    }

    @objc func tapOnDoneBut(){
        textField.resignFirstResponder()
        
    }

// inAppPurchase----------------------------------------------------------
    @IBAction func didTapPurchasePlans(){
        guard let vc = storyboard?.instantiateViewController(identifier: "upgrade") else {
            return
        }
        vc.title = "Purchase Plans"
        navigationController?.pushViewController(vc, animated: true)
    }
    
    
    @IBAction func ratingBtn(_ sender: UIButton) {
        ratingSheet(message: "지금 평가하시겠습니까?")
    }
    
    
    
    @IBAction func btn_URL(_ sender: UIButton) {
        urlAlert(message: "기부처 사이트를 여시겠습니까?")
    }
    
    
    @IBAction func ContactUs(_ sender: UIButton) {
        alert(message: "개발자의 메일 주소를 복사하시겠습니까?")
    }
// ----------------------------------------------------------
    

    
}
