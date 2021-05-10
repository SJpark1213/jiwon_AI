//
//  MainViewController.swift
//  SwithDog
//
//  Created by Leekyujin on 2020/10/21.
//  Copyright © 2020 Leekyujin. All rights reserved.
//

import UIKit

class MainViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        
        if UserDefaults.standard.bool(forKey: "PurchasePlans") {
            // 플랜 구매전 기능만 보여주기
        }
        else{
            // 구매후 기능 입력하기
        }
    }
    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
