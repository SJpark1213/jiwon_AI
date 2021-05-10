//
//  ExercisePlanScrollView.swift
//  Pages
//
//  Created by Leekyujin on 2020/11/13.
//

import UIKit
import FSCalendar


class ExercisePlanScrollView: UIScrollView {
    
    let scrollView = UIScrollView()
    let contentsView = UIView()
    
    let logoView: UIImageView = { () -> UIImageView in
        let imageView = UIImageView()
        
        imageView.contentMode = .scaleAspectFit
        imageView.image = UIImage.init(named: "Logo")
        return imageView
    }()
    
    let calendarView: UIView = { () -> UIView in
        let view: UIView = FSCalendar()
        
        
        return view
    }()
    
    let circularProgressView = CircularProgressBar()
    
    let planLabel: UILabel = { () -> UILabel in
        let label = UILabel()
        
        label.text = "현재 구독중인 플랜은 A 플랜입니다!"
        label.font = label.font.withSize(20)
        label.textAlignment = .center
        return label
    }()
        
    /*required init?(coder: NSCoder) {
            super.init(coder: coder)
    }*/
    
    func initView(){
       
        contentsView.addSubview(calendarView)
        contentsView.addSubview(circularProgressView)
        contentsView.addSubview(planLabel)
        scrollView.addSubview(contentsView)
        self.addSubview(scrollView)
     
        
        calendarView.trailingAnchor.constraint(equalTo: self.contentsView.trailingAnchor).isActive = true
        calendarView.leadingAnchor.constraint(equalTo: self.contentsView.leadingAnchor).isActive = true
        calendarView.topAnchor.constraint(equalTo: self.logoView.bottomAnchor, constant: 8).isActive = true
        calendarView.heightAnchor.constraint(equalToConstant: 250).isActive = true
        
        circularProgressView.trailingAnchor.constraint(equalTo: self.contentsView.trailingAnchor).isActive = true
        circularProgressView.leadingAnchor.constraint(equalTo: self.contentsView.leadingAnchor).isActive = true
        circularProgressView.topAnchor.constraint(equalTo: self.calendarView.bottomAnchor, constant: 8).isActive = true
        circularProgressView.heightAnchor.constraint(equalTo:  circularProgressView.widthAnchor).isActive = true
        
      
    }

    
    
}
