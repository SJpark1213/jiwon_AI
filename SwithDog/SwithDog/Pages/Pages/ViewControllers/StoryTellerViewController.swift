//
//  StoryTellerViewController.swift
//  Pages
//
//  Created by Leekyujin on 2020/11/13.
//

import UIKit

class StoryTellerViewController: UIViewController {

    let firstImage = UIImageView()
    let secondImage = UIImageView()
    let thirdImage = UIImageView()
    let fourthImage = UIImageView()
    
    let firstText = UILabel()
    let secondText = UILabel()
    let thirdText = UILabel()
    let fourthText = UILabel()
    
    
    override func viewDidLoad() {
        super.viewDidLoad()

        view.backgroundColor = .white
        // Do any additional setup after loading the view.
    }
    

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        let scrollView = UIScrollView()
        scrollView.frame = CGRect(x: 0, y: view.safeAreaInsets.top - 5, width: view.frame.size.width, height: view.frame.size.height)
        view.addSubview(scrollView)
        scrollView.backgroundColor = .clear
        
        
        // MARK: - firstImageView
        
        firstImage.image = UIImage(named: "illust 1")
        firstImage.backgroundColor = .white
        firstImage.contentMode = .scaleAspectFit
        firstImage.clipsToBounds = true
        firstImage.layer.cornerRadius = 40 / 2

        scrollView.addSubview(firstImage)
        
        firstImage.translatesAutoresizingMaskIntoConstraints = false
        firstImage.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 50).isActive = true
        firstImage.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        firstImage.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        firstImage.heightAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 1.12).isActive = true
        
        // first Text
        firstText.text = "한 해에 약 10만 마리의 강아지가\n유기견이 된다는 사실, 알고 계셨나요?"
        firstText.font = .systemFont(ofSize: 20, weight: .light)
        firstText.numberOfLines = 3
        firstText.textAlignment = .center
        
        scrollView.addSubview(firstText)
        firstText.translatesAutoresizingMaskIntoConstraints = false
        firstText.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 580).isActive = true
        firstText.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        
        
        // MARK: - secondImageView

        secondImage.image = UIImage(named: "illust 2")
        secondImage.backgroundColor = .white
        secondImage.clipsToBounds = true
        secondImage.layer.cornerRadius = 40 / 2

        scrollView.addSubview(secondImage)

        secondImage.translatesAutoresizingMaskIntoConstraints = false
        secondImage.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 700).isActive = true
        secondImage.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        secondImage.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        secondImage.heightAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 1.12).isActive = true

        
        // second Text
        secondText.text = "보호소의 열악한 상황으로 인해\n다섯마리 중 한마리는 안락사에 이르게 됩니다."
        secondText.font = .systemFont(ofSize: 20, weight: .light)
        secondText.numberOfLines = 3
        secondText.textAlignment = .center
        
        scrollView.addSubview(secondText)
        secondText.translatesAutoresizingMaskIntoConstraints = false
        secondText.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 1230).isActive = true
        secondText.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true

        // MARK: - thirdImage

        thirdImage.image = UIImage(named: "illust 3")
        thirdImage.backgroundColor = .white
        thirdImage.clipsToBounds = true
        thirdImage.layer.cornerRadius = 40 / 2

        scrollView.addSubview(thirdImage)

        thirdImage.translatesAutoresizingMaskIntoConstraints = false
        thirdImage.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 1350).isActive = true
        thirdImage.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        thirdImage.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        thirdImage.heightAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 1.12).isActive = true

        
        // third Text
        thirdText.text = "우리의 따뜻함이 모여\n강아지들에게 희망이 될 수 있다면\n얼마나 좋을까요?"
        thirdText.font = .systemFont(ofSize: 20, weight: .light)
        thirdText.numberOfLines = 3
        thirdText.textAlignment = .center
        
        scrollView.addSubview(thirdText)
        thirdText.translatesAutoresizingMaskIntoConstraints = false
        thirdText.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 1880).isActive = true
        thirdText.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        
        // MARK: - fourthImage

        fourthImage.image = UIImage(named: "illust 4")
        fourthImage.backgroundColor = .white
        fourthImage.clipsToBounds = true
        fourthImage.layer.cornerRadius = 40 / 2

        scrollView.addSubview(fourthImage)

        fourthImage.translatesAutoresizingMaskIntoConstraints = false
        fourthImage.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 2020).isActive = true
        fourthImage.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        fourthImage.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        fourthImage.heightAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 1.12).isActive = true

        
        // forth Text
        fourthText.text = "함께 운동하고,\n그 마음을 강아지들에게 전해 보세요."
        fourthText.font = .systemFont(ofSize: 20, weight: .light)
        fourthText.numberOfLines = 3
        fourthText.textAlignment = .center
        
        scrollView.addSubview(fourthText)
        fourthText.translatesAutoresizingMaskIntoConstraints = false
        fourthText.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 2560).isActive = true
        fourthText.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        
        
        //MARK:- scrollView contentSize
        scrollView.contentSize = CGSize(width: view.frame.size.width, height: 2720)
        }
        
 
}
