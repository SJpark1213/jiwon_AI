//
//  WalkthroughContentViewController.swift
//  Pages
//
//  Created by Leekyujin on 2020/11/20.
//

import UIKit

class WalkthroughContentViewController: UIViewController {

    //MARK: - outlets
    
    @IBOutlet var body: UILabel! {
        didSet{
            body.numberOfLines = 3
        }
    }
    
    @IBOutlet var storyImage: UIImageView!
    
    //MARK:- properties
    
    var index = 0
    var bodyText = ""
    var imageFile = ""
   
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        body.text = bodyText
        storyImage.image = UIImage(named: imageFile)
        
        
    }
    


}
