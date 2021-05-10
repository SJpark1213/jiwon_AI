//
//  WalkthroughViewController.swift
//  Pages
//
//  Created by Leekyujin on 2020/11/20.
//

import UIKit

class WalkthroughViewController: UIViewController, WalkthroughPageViewControllerDelegate {
    
    //MARK:- Outlets
    
    @IBOutlet var pageControl: UIPageControl!
   
    @IBOutlet var getStartedButton: UIButton! {
        didSet {
            getStartedButton.layer.cornerRadius = 20.0
            getStartedButton.layer.masksToBounds = true
            getStartedButton.isHidden = true
        }
    }
    @IBOutlet var SkipButton: UIButton!

    //MARK:- properties
    var walkthroughPageViewController: WalkthroughPageViewController?
    
    //MARK:- Actions
    
    @IBAction func skipButtonTapped(_ sender: UIButton) {
        guard let nextVC = self.storyboard?.instantiateViewController(identifier: "TabBar") as? TabBarController else {return}
        self.navigationController?.pushViewController(nextVC, animated: true)
    }
    
    
    
    @IBAction func getStratedButtonTapped(sender: UIButton) {
        if let index = walkthroughPageViewController?.currentIndex {
            switch index {
            case 0...2:
                walkthroughPageViewController?.forwardPage()
                
            case 3:
                guard let nextVC = self.storyboard?.instantiateViewController(identifier: "TabBar") as? TabBarController else {return}
                self.navigationController?.pushViewController(nextVC, animated: true)
           
            default:
                break
            }
        }
        updateUI()
    }
    
    func updateUI() {
        if let index = walkthroughPageViewController?.currentIndex {
            switch index {
            case 0...2:
                getStartedButton.isHidden = true
                SkipButton.isHidden = false
                
            case 3:
                getStartedButton.isHidden = false
                SkipButton.isHidden = true
           
            default:
                break
            }
            pageControl.currentPage = index
        }
    }
    
    func didUpdatePageIndex(currenIndex: Int) {
        updateUI()
    }
    //MARK:- view controller life cycle

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Do any additional setup after loading the view.
    }
    

    
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    
        let destination = segue.destination
        if let pageViewController = destination as? WalkthroughPageViewController {
            walkthroughPageViewController = pageViewController
            walkthroughPageViewController?.walktrhoughDelegate = self
        }
    }
    

}
