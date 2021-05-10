//
//  ViewController.swift
//  Pages
//
//  Created by Leekyujin on 2020/11/12.
//
import UIKit
import FSCalendar

class PlanViewController: UIViewController, UIGestureRecognizerDelegate, FSCalendarDataSource, FSCalendarDelegate {
    
    // MARK: - Variables
    let backgroundPattern = UIImageView()
    var endAngle: CGFloat = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .secondarySystemBackground
        
        backgroundPattern.image = UIImage(named: "backgroundPattern(gray)")
        backgroundPattern.contentMode = .scaleAspectFill
        backgroundPattern.clipsToBounds = true
        self.view.addSubview(backgroundPattern)
        
        backgroundPattern.translatesAutoresizingMaskIntoConstraints = false
        backgroundPattern.topAnchor.constraint(equalTo: view.topAnchor).isActive = true
        backgroundPattern.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 1).isActive = true
        backgroundPattern.bottomAnchor.constraint(equalTo: view.bottomAnchor).isActive = true
    
        
        //HistoryNavigationBackBarItemTitle
        let backBarButtonItem = UIBarButtonItem(title: "Plan", style: .plain, target: self, action: nil)
        self.navigationItem.backBarButtonItem = backBarButtonItem
        
        // MARK: - Update
        additionalFetch()
        updateComponents()
        
        // tapGesture
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(self.handleTap(_:)))
        tapGesture.delegate = self
        provisionalProgressBar.addGestureRecognizer(tapGesture)
        
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(true)
        updateComponents()
    }
    
//    override func viewDidAppear(_ animated: Bool) {
//        let storyboard = UIStoryboard(name: "Main", bundle: nil)
//        if let walkthroughViewController = storyboard.instantiateViewController(withIdentifier: "WalkthroughViewController") as? WalkthroughViewController {
//            present(walkthroughViewController, animated: true, completion: nil)
//        }
//    
//    }
    
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    
    // MARK: - objc Functions
    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        provisionalProgressBar.setProgress(to: 1, withAnimation: true)
        
        let today = NSDate()
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        let stringDate = formatter.string(from: today as Date)
        
        //lineProgressBar1.updateProgressBarView(labelText: "건강도", percent: Double((nowUser.pastSquatRepsDict[stringDate] ?? 0) / (planList[nowUser.currentPlan].squatRepsPerDay)), color: UIColor.red)
        lineProgressBar1.updateProgressBarView(labelText: "건강도",percent: Double(1000/(planList[nowUser.currentPlan].squatRepsPerDay)), color: UIColor.red)
        
        //lineProgressBar2.updateProgressBarView(labelText: "친밀도", percent: Double((nowUser.pastLungeRepsDict[stringDate] ?? 0) / (planList[nowUser.currentPlan].lungeRepsPerDay)), color: UIColor.yellow)
        lineProgressBar2.updateProgressBarView(labelText: "친밀도",percent: Double(2000/(planList[nowUser.currentPlan].lungeRepsPerDay)), color: UIColor.yellow)
        
        //lineProgressBar3.updateProgressBarView(labelText: "포만감", percent: Double((nowUser.pastJumpingJackRepsDict[stringDate] ?? 0) / (planList[nowUser.currentPlan].jumpingJackRepsPerDay)), color: UIColor.green)
        lineProgressBar3.updateProgressBarView(labelText: "포만감",percent: Double(3000/(planList[nowUser.currentPlan].jumpingJackRepsPerDay)), color: UIColor.green)
    }
    
    // MARK: - Functions
    func updateComponents(){
        let today = NSDate()
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        let stringDate = formatter.string(from: today as Date)
        
        // FSCalendar
        calendarView.delegate = self
        calendarView.dataSource = self
        
        // CalendarLabel
        calendarLabel.text = "달력에서 날짜를 골라주세요"
        
        // ProvisionalProgressBar
        let nowPlantotalReps = planList[nowUser.currentPlan].squatRepsPerDay + planList[nowUser.currentPlan].lungeRepsPerDay + planList[nowUser.currentPlan].jumpingJackRepsPerDay
        let todayTotalReps = (nowUser.pastSquatRepsDict[stringDate] ?? 0) + (nowUser.pastLungeRepsDict[stringDate] ?? 0) + (nowUser.pastJumpingJackRepsDict[stringDate] ?? 0)
        //endAngle = CGFloat(todayTotalReps/nowPlantotalReps)
        endAngle = CGFloat(CGFloat.pi*2)
        provisionalProgressBar.setMyEndAngle(endAngle)
        
        // 3 ProgressBars
        
        // Plan Achievement Label
        let nowPlanName = planList[nowUser.currentPlan].planName
        if nowUser.currentPlan == 0 {
            planAchieveButton.text = "현재 이용 중인 플랜은 \n \(nowPlanName)입니다."
        }else {
            planAchieveButton.text = "\(nowUser.currentPlanDaysList[0])~\(nowUser.currentPlanDaysList[nowUser.currentPlanDaysList.count - 1]) \n 현재 이용 중인 플랜은 \n \(nowPlanName)입니다."
        }
    }
    
    func calendar(_ calendar: FSCalendar, didSelect date: Date, at monthPosition: FSCalendarMonthPosition) {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        let stringDate = formatter.string(from: date)
        
        if let squatReps = nowUser.pastSquatRepsDict[stringDate], let lungeReps = nowUser.pastLungeRepsDict[stringDate], let jumpingJackReps = nowUser.pastJumpingJackRepsDict[stringDate]{
            let squatPercent = squatReps / planList[nowUser.currentPlan].squatRepsPerDay
            let lungePercent = lungeReps / planList[nowUser.currentPlan].lungeRepsPerDay
            let jumpingJackPercent = jumpingJackReps / planList[nowUser.currentPlan].jumpingJackRepsPerDay
            
            calendarLabel.text = "\(stringDate) \n 스쿼트: \(squatPercent)% 런지: \(lungePercent)% 팔벌려뛰기: \(jumpingJackPercent)%"

        }else{
            calendarLabel.text = "운동하지 않은 날입니다"
        }
    }

//    func calendar(_ calendar: FSCalendar, numberOfEventsFor date: Date) -> Int {
//        let formatter = DateFormatter()
//        formatter.dateFormat = "yyyy-MM-dd"
//        let dateString = formatter.string(from: date)
//
//        //let exDateList: [Date] = [formatter.date(from: "2020-11-09")!]
//        
//        if nowUser.currentPlanDaysList.contains(dateString){
//            return 1
//        }else{
//            return 0
//        }
//    }
    
    func calendar(_ calendar: FSCalendar, imageFor date: Date) -> UIImage? {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        let dateString = formatter.string(from: date)

        let image = UIImage(named: "dogPaw")
        let newImage = image?.resizeImage(10, opaque: false)

        if nowUser.currentPlanDaysList.contains(dateString){
            return newImage
        }

        return nil
    }
    
    // MARK: - AutoLayout
    let calendarTitleLabel = UILabel()
    let calendarBackgroundView = UIView()
    let calendarView = FSCalendar()
    let calendarLabel = UILabel()
    
    let titleLabel = UILabel()
    let historyButton = UILabel()
    var provisionalProgressBar = CircularProgressBar()
    var lineProgressBar1 = LineProgressBarView()
    var lineProgressBar2 = LineProgressBarView()
    var lineProgressBar3 = LineProgressBarView()
    
    let planAchieveButton = UILabel()
    let planViewCopyrightsLabel = UILabel()

    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        // MARK: - scrollView
        let scrollView = UIScrollView()
        scrollView.frame = CGRect(x: 0, y: view.safeAreaInsets.top - 5, width: view.frame.size.width, height: view.frame.size.height)
        view.addSubview(scrollView)
        scrollView.backgroundColor = .clear
        
        // MARK: - Calendar Background
        scrollView.addSubview(calendarBackgroundView)
        
        calendarBackgroundView.layer.cornerRadius = 40 / 2
        calendarBackgroundView.translatesAutoresizingMaskIntoConstraints = false
        calendarBackgroundView.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 60).isActive = true
        calendarBackgroundView.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        calendarBackgroundView.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        calendarBackgroundView.heightAnchor.constraint(equalTo: calendarBackgroundView.widthAnchor, multiplier: 0.7).isActive = true
        calendarBackgroundView.backgroundColor = .white
        
        // MARK: - CalendarView
        scrollView.addSubview(calendarView)
        
        calendarView.layer.cornerRadius = 40 / 2
        calendarView.translatesAutoresizingMaskIntoConstraints = false
        calendarView.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 60).isActive = true
        calendarView.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        calendarView.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        calendarView.heightAnchor.constraint(equalTo: calendarView.widthAnchor, multiplier: 0.7).isActive = true
        
        // MARK: - CalendarLabel
        scrollView.addSubview(calendarLabel)
        
        calendarLabel.layer.cornerRadius = 30 / 2
        calendarLabel.clipsToBounds = true
        
        calendarLabel.translatesAutoresizingMaskIntoConstraints = false
        calendarLabel.topAnchor.constraint(equalTo: calendarView.bottomAnchor, constant: 10).isActive = true
        calendarLabel.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        calendarLabel.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        calendarLabel.heightAnchor.constraint(equalToConstant: 60).isActive = true
        calendarLabel.backgroundColor = .white
        
        calendarLabel.textAlignment = .center
        calendarLabel.font = UIFont.boldSystemFont(ofSize: 15)
        calendarLabel.numberOfLines = 3
        
        // MARK: - historyBtn
        
        historyButton.backgroundColor = #colorLiteral(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        historyButton.clipsToBounds = true
        historyButton.layer.cornerRadius = 50 / 2

        scrollView.addSubview(historyButton)
        
        let historyButtonHeight: CGFloat = 180
        
        historyButton.translatesAutoresizingMaskIntoConstraints = false
        historyButton.heightAnchor.constraint(equalToConstant: historyButtonHeight).isActive = true
        historyButton.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        historyButton.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        historyButton.topAnchor.constraint(equalTo: calendarLabel.bottomAnchor, constant: 60).isActive = true
        
        //historyButton.addTarget(self, action: #selector(historyBtn), for: .touchUpInside)
        
        // MARK: - provisionalProgressBar
        scrollView.addSubview(provisionalProgressBar)
        
        let provisionalProgressBarX: CGFloat = (scrollView.frame.width * 0.1 )/2 + 20
        let provisionalProgressBarY: CGFloat = (scrollView.frame.width * 0.63 + 125) + 90
        
        provisionalProgressBar.translatesAutoresizingMaskIntoConstraints = false
        provisionalProgressBar.frame = CGRect(x: provisionalProgressBarX, y: provisionalProgressBarY, width: historyButtonHeight - 40, height: historyButtonHeight - 40)
        
        // MARK: - Three ProgressBars
        scrollView.addSubview(lineProgressBar1)
        scrollView.addSubview(lineProgressBar2)
        scrollView.addSubview(lineProgressBar3)

        lineProgressBar1.translatesAutoresizingMaskIntoConstraints = false
        lineProgressBar2.translatesAutoresizingMaskIntoConstraints = false
        lineProgressBar3.translatesAutoresizingMaskIntoConstraints = false

        lineProgressBar1.updateSizeofView(labelText: "건강도",frame: CGRect(x: provisionalProgressBar.frame.maxX + 20, y: provisionalProgressBar.frame.minY, width: scrollView.frame.width * 0.9 - provisionalProgressBar.frame.width - 60, height: provisionalProgressBar.frame.height/3))
        lineProgressBar2.updateSizeofView(labelText: "친밀도",frame: CGRect(x: provisionalProgressBar.frame.maxX + 20, y: provisionalProgressBar.frame.minY + provisionalProgressBar.frame.height/3, width: scrollView.frame.width * 0.9 - provisionalProgressBar.frame.width - 60, height: provisionalProgressBar.frame.height/3))
        lineProgressBar3.updateSizeofView(labelText: "포만감",frame: CGRect(x: provisionalProgressBar.frame.maxX + 20, y: provisionalProgressBar.frame.minY + provisionalProgressBar.frame.height * 2/3, width: scrollView.frame.width * 0.9 - provisionalProgressBar.frame.width - 60, height: provisionalProgressBar.frame.height/3))
        
        // MARK: - planAchieveBtn
//        planAchieveButton.setImage(UIImage (systemName: "chart.bar"), for: .normal)
//        planAchieveButton.setImage(UIImage (systemName: "chart.bar.fill"), for: .highlighted)
//        planAchieveButton.imageView?.contentMode = .scaleToFill
        planAchieveButton.textAlignment = .center
        planAchieveButton.font = UIFont.boldSystemFont(ofSize: 25)
        planAchieveButton.clipsToBounds = true
        planAchieveButton.numberOfLines = 3
        planAchieveButton.backgroundColor = #colorLiteral(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        planAchieveButton.layer.cornerRadius = 50 / 2

        scrollView.addSubview(planAchieveButton)
        
        planAchieveButton.translatesAutoresizingMaskIntoConstraints = false
        planAchieveButton.heightAnchor.constraint(equalToConstant: 120).isActive = true
        planAchieveButton.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        planAchieveButton.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        planAchieveButton.topAnchor.constraint(equalTo: historyButton.bottomAnchor, constant: 15).isActive = true
        
        //planAchieveButton.addTarget(self, action: #selector(planAchieveBtn), for: .touchUpInside)
        
        // MARK: - calendarTitleLabel
        calendarTitleLabel.text = "기록"
        calendarTitleLabel.backgroundColor = .clear
        calendarTitleLabel.font = UIFont.boldSystemFont(ofSize: 20)
        calendarTitleLabel.textAlignment = .left
        
        scrollView.addSubview(calendarTitleLabel)
        
        calendarTitleLabel.translatesAutoresizingMaskIntoConstraints = false
        calendarTitleLabel.leftAnchor.constraint(equalTo: calendarView.leftAnchor, constant: 7).isActive = true
        calendarTitleLabel.bottomAnchor.constraint(equalTo: calendarView.topAnchor, constant: -5).isActive = true
        
        // MARK: - titleLabel
        titleLabel.text = "성취"
        titleLabel.backgroundColor = .clear
        titleLabel.font = UIFont.boldSystemFont(ofSize: 20)
        titleLabel.textAlignment = .left
        
        scrollView.addSubview(titleLabel)
        
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        titleLabel.leftAnchor.constraint(equalTo: historyButton.leftAnchor, constant: 7).isActive = true
        titleLabel.bottomAnchor.constraint(equalTo: historyButton.topAnchor, constant: -5).isActive = true
        
        
        scrollView.contentSize = CGSize(width: view.frame.size.width, height: 900)
        
        
        // MARK: - copyrightsLabel
        planViewCopyrightsLabel.text = "© 2020.SwithDog.All rights reserved"
        planViewCopyrightsLabel.font = planViewCopyrightsLabel.font.withSize(11)
        planViewCopyrightsLabel.textColor = .lightGray
        scrollView.addSubview(planViewCopyrightsLabel)
        
        planViewCopyrightsLabel.translatesAutoresizingMaskIntoConstraints = false
        planViewCopyrightsLabel.bottomAnchor.constraint(equalTo: planAchieveButton.bottomAnchor, constant: 99).isActive = true
        planViewCopyrightsLabel.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        
    }
    
//    @objc func historyBtn() {
//
//        let button = dailyHistoryTableViewController()
//
//        self.navigationController?.pushViewController(button, animated: true)
        
        
        
//    }
//
//    @objc func planAchieveBtn() {
//        let button = PlanAchieveTableViewController()
//
//        self.navigationController?.pushViewController(button, animated: true)
//    }
//
//
//}
}

extension UIImage {
    func resizeImage(_ dimension: CGFloat, opaque: Bool, contentMode: UIView.ContentMode = .scaleAspectFit) -> UIImage {
            var width: CGFloat
            var height: CGFloat
            var newImage: UIImage

            let size = self.size
            let aspectRatio =  size.width/size.height

            switch contentMode {
                case .scaleAspectFit:
                    if aspectRatio > 1 {                            // Landscape image
                        width = dimension
                        height = dimension / aspectRatio
                    } else {                                        // Portrait image
                        height = dimension
                        width = dimension * aspectRatio
                    }

            default:
                fatalError("UIIMage.resizeToFit(): FATAL: Unimplemented ContentMode")
            }

            if #available(iOS 10.0, *) {
                let renderFormat = UIGraphicsImageRendererFormat.default()
                renderFormat.opaque = opaque
                let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: renderFormat)
                newImage = renderer.image {
                    (context) in
                    self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
                }
            } else {
                UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), opaque, 0)
                    self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
                    newImage = UIGraphicsGetImageFromCurrentImageContext()!
                UIGraphicsEndImageContext()
            }

            return newImage
        }
}
