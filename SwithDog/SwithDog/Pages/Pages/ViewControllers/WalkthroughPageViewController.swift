//
//  WalkthroughPageViewController.swift
//  Pages
//
//  Created by Leekyujin on 2020/11/20.
//

import UIKit

protocol WalkthroughPageViewControllerDelegate: class {
    func didUpdatePageIndex(currenIndex: Int)
}

class WalkthroughPageViewController: UIPageViewController, UIPageViewControllerDataSource, UIPageViewControllerDelegate {

    //MARk:- properties
    
    weak var walktrhoughDelegate: WalkthroughPageViewControllerDelegate?
    
    var pageBodys = ["한 해에 약 10만 마리의 강아지가\n유기견이 된다는 사실, 알고 계셨나요?", "보호소의 열악한 상황으로 인해\n다섯마리 중 한마리는 안락사에 이르게 됩니다.", "우리의 따뜻함이 모여\n강아지들에게 희망이 될 수 있다면\n얼마나 좋을까요?", "함께 운동하고,\n그 마음을 강아지들에게 전해 보세요."]
    var pageImages = ["onBoarding 1", "onBoarding 2", "onBoarding 3", "onBoarding 4"]
    
    var currentIndex = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //Set the data source
        dataSource = self
        delegate = self
        
        //Create the first walkthrough screen
        if let startingViewController = contentViewController(at: 0) {
            setViewControllers([startingViewController], direction: .forward, animated: true, completion: nil)
        }
        

        // Do any additional setup after loading the view.
    }
    
    //MARK:- Page View Controller Data Source
    func pageViewController(_ pageViewController: UIPageViewController, viewControllerBefore viewController: UIViewController) -> UIViewController? {
        var index = (viewController as! WalkthroughContentViewController).index
        index -= 1
        
        return  contentViewController(at: index)
    }
    func pageViewController(_ pageViewController: UIPageViewController, viewControllerAfter viewController: UIViewController) -> UIViewController? {
        var index = (viewController as! WalkthroughContentViewController).index
        index += 1
        
        return  contentViewController(at: index)
    }
    
    //MARK:- Helper
    
    func contentViewController(at index: Int) -> WalkthroughContentViewController? {
        if index < 0 || index >= pageBodys.count {
            return nil
            
        }
        //Create a new View and pass suitabel Data
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        if let pageContentViewController = storyboard.instantiateViewController(identifier: "WalkthroughContentViewController") as? WalkthroughContentViewController {
            pageContentViewController.imageFile = pageImages[index]
            pageContentViewController.bodyText = pageBodys[index]
            
            pageContentViewController.index = index
            
            return pageContentViewController
        }
        
        return nil
    }
    
    func forwardPage() {
        currentIndex += 1
        if let nextViewController = contentViewController(at: currentIndex) {
            setViewControllers([nextViewController], direction: .forward, animated: true, completion: nil)
        }
    }
    
    //MARK:- page View controller Delegate
    
    func pageViewController(_ pageViewController: UIPageViewController, didFinishAnimating finished: Bool, previousViewControllers: [UIViewController], transitionCompleted completed: Bool) {
        if completed {
            if let contentViewController = pageViewController.viewControllers?.first as? WalkthroughContentViewController {
                currentIndex = contentViewController.index
                
                walktrhoughDelegate?.didUpdatePageIndex(currenIndex: currentIndex)
            }
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
