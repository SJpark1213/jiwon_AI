//
//  ViewController.swift
//  Pages
//
//  Created by Leekyujin on 2020/11/12.
//

import UIKit

struct CustomData {
    var image: UIImage = #imageLiteral(resourceName: "dogPawLogo")
}

class ViewController: UIViewController {
    

    var data = [
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo")),
        CustomData(image: #imageLiteral(resourceName: "dogPawLogo"))
    ]
    
    
    let backgroundPattern = UIImageView()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .secondarySystemBackground
        
        // MARK: - Update
        additionalFetch()
        
        
        backgroundPattern.image = UIImage(named: "backgroundPattern(gray)")
        backgroundPattern.contentMode = .scaleAspectFill
        backgroundPattern.clipsToBounds = true
        self.view.addSubview(backgroundPattern)
        
        backgroundPattern.translatesAutoresizingMaskIntoConstraints = false
        backgroundPattern.topAnchor.constraint(equalTo: view.topAnchor).isActive = true
        backgroundPattern.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 1).isActive = true
        backgroundPattern.bottomAnchor.constraint(equalTo: view.bottomAnchor).isActive = true
        
        
       
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(true)
        updateCollectionView()
    }
    
    let badgeShelf = UIImageView()
    
    let storeButton = UIButton()
    let lockButton = UIButton()
    

    let imageView = UIImageView()
    
    let profileFill = UIButton()
    let plusSign = UIImageView()
    let imageBackground = UIImageView()
    
    let storyTeller = UIButton()
    let imageBoard = UIImageView()
    let storyTitleBar = UILabel()
    let storyTellerTitleLabel = UILabel()
    let storySummary = UILabel()

    let websiteLabel = UILabel()
    let moveWebsiteButton = UIButton()
    
    let contactUsLebel = UILabel()
    let contactUsButton = UIButton()
    
    
    let copyrightsLabel = UILabel()
    
    let collectionViewBackground = UILabel()
    fileprivate let collectionView: UICollectionView = {
        let layout = UICollectionViewFlowLayout()
        layout.scrollDirection = .horizontal
        let cv = UICollectionView(frame: .zero, collectionViewLayout: layout)
        cv.translatesAutoresizingMaskIntoConstraints = false
        cv.register(CustomCell.self, forCellWithReuseIdentifier: "cell")
        
        return cv
    }()

    func updateCollectionView() {
        //Dog Image switch
        
        //MARK: - JiwonAdd
        let number = nowUser.graduatePlanList.count
        if number == 0{
            data[0] = CustomData(image: planList[0].planImage)
        }else{
            for i in 0...number-1{
                data[number-1-i] = CustomData(image: planList[nowUser.graduatePlanList[i]].planImage)
            }
        }
        collectionView.reloadData()
    }
        
        
        
//        if nowUser.currentPlan == 0 {
//            data.insert(CustomData(image: #imageLiteral(resourceName: "free plan icon")), at: 0)
//        }else {
//            data.insert(CustomData(image: .init(imageLiteralResourceName: "\(nowPlanImage)")), at: 0)
//        }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
      
        let scrollView = UIScrollView()
        scrollView.frame = CGRect(x: 0, y: view.safeAreaInsets.top - 5, width: view.frame.size.width, height: view.frame.size.height)
        view.addSubview(scrollView)
        scrollView.backgroundColor = .clear
        
        // MARK: - imageViewBackgroundColor
        imageBackground.layer.cornerRadius = 150 / 2
        imageBackground.backgroundColor = .white
        imageBackground.clipsToBounds = true

        
        scrollView.addSubview(imageBackground)
        
        imageBackground.translatesAutoresizingMaskIntoConstraints = false
        imageBackground.heightAnchor.constraint(equalToConstant: 150).isActive = true
        imageBackground.widthAnchor.constraint(equalToConstant: 150).isActive = true
        imageBackground.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        imageBackground.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 20).isActive = true
        
        // MARK: - imgageViewPlusSign
        plusSign.layer.cornerRadius = 40 / 2
        plusSign.backgroundColor = .clear
        plusSign.clipsToBounds = true
        plusSign.image = #imageLiteral(resourceName: "unnamed-2")

        
        scrollView.addSubview(plusSign)
        
        plusSign.translatesAutoresizingMaskIntoConstraints = false
        plusSign.heightAnchor.constraint(equalToConstant: 40).isActive = true
        plusSign.widthAnchor.constraint(equalToConstant: 40).isActive = true
        plusSign.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        plusSign.centerYAnchor.constraint(equalTo: imageBackground.centerYAnchor).isActive = true
        
        // MARK: - imageViewProfileField
        imageView.layer.cornerRadius = 150 / 2
        imageView.backgroundColor = .clear
        imageView.clipsToBounds = true
        
        scrollView.addSubview(imageView)
        
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.heightAnchor.constraint(equalToConstant: 150).isActive = true
        imageView.widthAnchor.constraint(equalToConstant: 150).isActive = true
        imageView.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        imageView.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 20).isActive = true
        
        
        // MARK: - profilePhotoSelectButton
        profileFill.imageView?.contentMode = .scaleToFill
        profileFill.backgroundColor = .clear
        profileFill.layer.cornerRadius = 150 / 2
        

        scrollView.addSubview(profileFill)
        
        profileFill.translatesAutoresizingMaskIntoConstraints = false
        profileFill.heightAnchor.constraint(equalToConstant: 150).isActive = true
        profileFill.widthAnchor.constraint(equalToConstant: 150).isActive = true
        profileFill.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        profileFill.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 20).isActive = true
        
        profileFill.addTarget(self, action: #selector(profileBtn), for: .touchUpInside)
        
        
        // MARK: - badge shelf collectionView
        
        scrollView.addSubview(collectionViewBackground)
        collectionViewBackground.backgroundColor = #colorLiteral(red: 0.2549019754, green: 0.2745098174, blue: 0.3019607961, alpha: 1)
        collectionViewBackground.alpha = 0.3
        collectionViewBackground.clipsToBounds = true
        collectionViewBackground.layer.cornerRadius = 35
        collectionViewBackground.translatesAutoresizingMaskIntoConstraints = false
        
        collectionViewBackground.heightAnchor.constraint(equalToConstant: 70).isActive = true
        collectionViewBackground.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        collectionViewBackground.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        collectionViewBackground.topAnchor.constraint(equalTo: profileFill.bottomAnchor, constant: 40).isActive = true
        
        // MARK: - badge collectionView
        scrollView.addSubview(collectionView)
        collectionView.backgroundColor = .clear
        collectionView.layer.cornerRadius = 35
        collectionView.heightAnchor.constraint(equalToConstant: 70).isActive = true
        collectionView.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        collectionView.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        collectionView.topAnchor.constraint(equalTo: profileFill.bottomAnchor, constant: 40).isActive = true
        
        collectionView.delegate = self
        collectionView.dataSource = self
        
//        // MARK: - badgeShelf
//
//        badgeShelf.backgroundColor = #colorLiteral(red: 0.2549019754, green: 0.2745098174, blue: 0.3019607961, alpha: 1)
//        badgeShelf.layer.cornerRadius = 70 / 2
//        badgeShelf.alpha = 0.3
//
//
//        scrollView.addSubview(badgeShelf)
//
//        badgeShelf.translatesAutoresizingMaskIntoConstraints = false
//        badgeShelf.heightAnchor.constraint(equalToConstant: 70).isActive = true
//        badgeShelf.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
//        badgeShelf.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
//        badgeShelf.topAnchor.constraint(equalTo: profileFill.bottomAnchor, constant: 40).isActive = true
        
        
        // MARK: - lockButton
        
//        lockButton.setImage(UIImage (systemName: "lock.circle"), for: .normal)
//        lockButton.setImage(UIImage (systemName: "lock.circle.fill"), for: .highlighted)
//        lockButton.backgroundColor = .clear
//        lockButton.layer.cornerRadius = 70 / 2
//
//
//
//        scrollView.addSubview(lockButton)
//
//        lockButton.translatesAutoresizingMaskIntoConstraints = false
//        lockButton.heightAnchor.constraint(equalToConstant: 70).isActive = true
//        lockButton.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
//        lockButton.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
//        lockButton.topAnchor.constraint(equalTo: profileFill.bottomAnchor, constant: 40).isActive = true
//
//        lockButton.addTarget(self, action: #selector(lockBtn), for: .touchUpInside)
        
        
        // MARK: - storeBtn
        storeButton.setImage(UIImage (systemName: "cart.badge.plus"), for: .normal)
        storeButton.setImage(UIImage (systemName: "cart.fill.badge.plus"), for: .highlighted)
        storeButton.imageView?.contentMode = .scaleAspectFit
        storeButton.backgroundColor = #colorLiteral(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        storeButton.layer.cornerRadius = 50 / 2


        scrollView.addSubview(storeButton)
        
        
        storeButton.translatesAutoresizingMaskIntoConstraints = false
        storeButton.heightAnchor.constraint(equalToConstant: 50).isActive = true
        storeButton.widthAnchor.constraint(equalToConstant: 50).isActive = true
        storeButton.topAnchor.constraint(equalTo: scrollView.topAnchor, constant: 10).isActive = true
        storeButton.rightAnchor.constraint(equalTo: collectionView.rightAnchor, constant: 0).isActive = true
        
        storeButton.addTarget(self, action: #selector(storeBtn), for: .touchUpInside)
        
        
        
        
        
        
        // MARK: - storyTeller
//        storyTeller.setImage(UIImage (systemName: "chart.bar"), for: .normal)
//        storyTeller.setImage(UIImage (systemName: "chart.bar.fill"), for: .highlighted)
        storyTeller.imageView?.contentMode = .scaleToFill
        storyTeller.backgroundColor = #colorLiteral(red: 1, green: 1, blue: 1, alpha: 1)
        storyTeller.layer.cornerRadius = 50 / 2

        scrollView.addSubview(storyTeller)
        
        storyTeller.translatesAutoresizingMaskIntoConstraints = false
        storyTeller.heightAnchor.constraint(equalToConstant: 330).isActive = true
        storyTeller.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        storyTeller.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        storyTeller.topAnchor.constraint(equalTo: collectionView.bottomAnchor, constant: 60).isActive = true
        
        storyTeller.addTarget(self, action: #selector(storyTellerBtn), for: .touchUpInside)
        
        
        // MARK: - imageBoard
        
        imageBoard.image = UIImage(named: "illust 1")
        imageBoard.backgroundColor = #colorLiteral(red: 0.9529411793, green: 0.6862745285, blue: 0.1333333403, alpha: 1)
        imageBoard.contentMode = .scaleAspectFill
        imageBoard.clipsToBounds = true
        imageBoard.roundCornerView(corners: [.topLeft, .topRight], radius: 40 / 2)

        scrollView.addSubview(imageBoard)
        
        imageBoard.translatesAutoresizingMaskIntoConstraints = false
        imageBoard.topAnchor.constraint(equalTo: collectionView.bottomAnchor, constant: 50).isActive = true
        imageBoard.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        imageBoard.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        imageBoard.heightAnchor.constraint(equalToConstant: 260).isActive = true
        
        
        // MARK: - storyTitleBar
        storyTitleBar.text = "  유기견에 대하여"
        storyTitleBar.textColor = .black
        storyTitleBar.backgroundColor = .clear
        storyTitleBar.textAlignment = .left
        storyTitleBar.font = storyTitleBar.font.withSize(25)
        storyTitleBar.font = UIFont.boldSystemFont(ofSize: 25)
        storyTitleBar.roundCornerView(corners: [.bottomLeft, .bottomRight], radius: 40 / 2)
        
        scrollView.addSubview(storyTitleBar)
        
        storyTitleBar.translatesAutoresizingMaskIntoConstraints = false
        storyTitleBar.topAnchor.constraint(equalTo: imageBoard.bottomAnchor, constant: 5).isActive = true
        storyTitleBar.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        storyTitleBar.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        storyTitleBar.heightAnchor.constraint(equalToConstant: 40).isActive = true
        
        // MARK: - storySummary
        storySummary.text = "   우리가 몰랐던 유기견에 대한 이야기"
        storySummary.textColor = .gray
        storySummary.backgroundColor = .clear
        storySummary.textAlignment = .left
        storySummary.font = storySummary.font.withSize(17)
        storySummary.numberOfLines = 3
        
        scrollView.addSubview(storySummary)
        
        storySummary.translatesAutoresizingMaskIntoConstraints = false
        storySummary.topAnchor.constraint(equalTo: storyTitleBar.bottomAnchor, constant: -10).isActive = true
        storySummary.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.9).isActive = true
        storySummary.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        storySummary.heightAnchor.constraint(equalToConstant: 40).isActive = true
        
        
        
        // MARK: - moveWebsiteBtn
        moveWebsiteButton.setTitle("기부처 연결하기", for: .normal)
        moveWebsiteButton.setTitleColor(.systemBlue, for: .normal)
        moveWebsiteButton.backgroundColor = #colorLiteral(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        moveWebsiteButton.layer.cornerRadius = 40 / 2


        scrollView.addSubview(moveWebsiteButton)
        
        moveWebsiteButton.translatesAutoresizingMaskIntoConstraints = false
        moveWebsiteButton.heightAnchor.constraint(equalToConstant: 60).isActive = true
        moveWebsiteButton.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.42).isActive = true
        moveWebsiteButton.leftAnchor.constraint(equalTo: storyTeller.leftAnchor, constant: 0).isActive = true
        moveWebsiteButton.topAnchor.constraint(equalTo: storyTeller.bottomAnchor, constant: 40).isActive = true
        
        moveWebsiteButton.addTarget(self, action: #selector(moveWebsiteBtn), for: .touchUpInside)
        
        

        
        
        
        // MARK: - contactUsBtn
        contactUsButton.setTitle("contact us", for: .normal)
        contactUsButton.setTitleColor(.systemBlue, for: .normal)
        contactUsButton.backgroundColor = #colorLiteral(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        contactUsButton.layer.cornerRadius = 40 / 2


        scrollView.addSubview(contactUsButton)
        
        contactUsButton.translatesAutoresizingMaskIntoConstraints = false
        contactUsButton.heightAnchor.constraint(equalToConstant: 60).isActive = true
        contactUsButton.widthAnchor.constraint(equalTo: scrollView.widthAnchor, multiplier: 0.42).isActive = true
        contactUsButton.rightAnchor.constraint(equalTo: storyTeller.rightAnchor, constant: 0).isActive = true
        contactUsButton.topAnchor.constraint(equalTo: storyTeller.bottomAnchor, constant: 40).isActive = true
        
        contactUsButton.addTarget(self, action: #selector(contactUsBtn), for: .touchUpInside)
        
        

        
        // MARK: - copyrightsLabel
        copyrightsLabel.text = "© 2020.SwithDog.All rights reserved"
        copyrightsLabel.font = copyrightsLabel.font.withSize(11)
        copyrightsLabel.textColor = .lightGray
        scrollView.addSubview(copyrightsLabel)
        
        copyrightsLabel.translatesAutoresizingMaskIntoConstraints = false
        copyrightsLabel.bottomAnchor.constraint(equalTo: contactUsButton.bottomAnchor, constant: 50).isActive = true
        copyrightsLabel.centerXAnchor.constraint(equalTo: scrollView.centerXAnchor).isActive = true
        
        // MARK: - StoryTellerTitleLabel
        storyTellerTitleLabel.text = "우리가 몰랐던 이야기"
        storyTellerTitleLabel.backgroundColor = .clear
        storyTellerTitleLabel.font = UIFont.boldSystemFont(ofSize: 20)
        storyTellerTitleLabel.textAlignment = .left
        
        scrollView.addSubview(storyTellerTitleLabel)
        
        storyTellerTitleLabel.translatesAutoresizingMaskIntoConstraints = false
        storyTellerTitleLabel.leftAnchor.constraint(equalTo: imageBoard.leftAnchor, constant: 7).isActive = true
        storyTellerTitleLabel.bottomAnchor.constraint(equalTo: imageBoard.topAnchor, constant: -5).isActive = true
        
        
        
        scrollView.contentSize = CGSize(width: view.frame.size.width, height: 910)
        
        
    }
    
    @objc func profileBtn() {
        
        showImagePickerControllerActionSheet()
        
        
    }
    
    
    @objc func storeBtn() {
        
        let button = StoreViewController()
        present(button, animated: true, completion: nil)
        
    }
    
    //MARK:- lock button
//    @objc func lockBtn() {
//
//        let alert = UIAlertController(title: "구매 후 이용할 수 있는 상품입니다", message: "구매를 위해 상점으로 이동하시겠습니까?", preferredStyle: .alert)
//        let vc = StoreViewController()
//        alert.addAction(UIAlertAction(title: "네", style: .default, handler: { (action: UIAlertAction!) in self.present(vc, animated: true, completion: nil)}))
//        alert.addAction(UIAlertAction(title: "아니오", style: .destructive, handler: nil))
//
//        present(alert, animated: true, completion: nil)
//
//    }
  
   
    @objc func moveWebsiteBtn() {
        
        let alert = UIAlertController(title: "사파리로 이동", message: "기부처 사이트로 이동하시겠습니까?", preferredStyle: .alert)
        let okAction = UIAlertAction(title: "확인", style: .default) {(action) in
            UIApplication.shared.open(NSURL(string: "http://www.animal.go.kr")! as URL)
        }
        let cancelAction = UIAlertAction(title: "취소", style: .destructive, handler: nil)
        alert.addAction(okAction)
        alert.addAction(cancelAction)
        
        present(alert, animated: true, completion: nil)
        
    }

    
    @objc func contactUsBtn() {
        
        let alert = UIAlertController(title: "개발자와 연결하기", message: "개발자의 이메일 주소를 복사하시겠습니까?", preferredStyle: .alert)
        
        let okAction = UIAlertAction(title: "확인", style: .default) {(copy) in
            UIPasteboard.general.string = "sto-leege@naver.com"
        }
        let cancelAction = UIAlertAction(title: "취소", style: .destructive, handler: nil)
        alert.addAction(okAction)
        alert.addAction(cancelAction)
        
        present(alert, animated: true, completion: nil)
        
    }

    @objc private func storyTellerBtn() {
        let button = StoryTellerViewController()
        button.title = "유기견 이야기"
        let nVC = UINavigationController(rootViewController: button)
        
        present(nVC, animated: true, completion: nil)
    }
}


extension UIView {
    
    func roundCornerView(corners: UIRectCorner, radius: CGFloat) {
        let path = UIBezierPath(roundedRect: self.bounds, byRoundingCorners: corners, cornerRadii: .init(width: radius, height: radius))
        let mask = CAShapeLayer()
        mask.path = path.cgPath
        self.layer.mask = mask
        
    }
    
}


extension ViewController: UICollectionViewDelegateFlowLayout, UICollectionViewDataSource {
    public func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {
        return CGSize(width: 68, height: 68)
    }
    
    public func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        
        return 20
    }
    
    public func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "cell", for: indexPath) as! CustomCell
        cell.backgroundColor = .clear
        cell.clipsToBounds = true
        cell.layer.cornerRadius = 34
        cell.data = self.data[indexPath.row]
        
        return cell
    }
  //MARK:- cell select action, connect storeView.
//    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
//        let alert = UIAlertController(title: "구매 후 이용할 수 있는 상품입니다", message: "구매를 위해 상점으로 이동하시겠습니까?", preferredStyle: .alert)
//        let vc = StoreViewController()
//        alert.addAction(UIAlertAction(title: "네", style: .default, handler: { (action: UIAlertAction!) in self.present(vc, animated: true, completion: nil)}))
//        alert.addAction(UIAlertAction(title: "아니오", style: .destructive, handler: nil))
//
//        present(alert, animated: true, completion: nil)
//    }
}

class CustomCell: UICollectionViewCell {
    
    var data: CustomData? {
        didSet{
            guard let data = data else{return}
            bg.image = data.image
        }

    }
    
    fileprivate let bg: UIImageView = {
        let iv = UIImageView()
        iv.image = #imageLiteral(resourceName: "unnamed-2")
        iv.translatesAutoresizingMaskIntoConstraints = false
        iv.contentMode = .scaleAspectFit
        iv.clipsToBounds = true
        
        return iv
    }()
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        
        contentView.addSubview(bg)
        bg.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 8).isActive = true
        bg.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 8).isActive = true
        bg.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -8).isActive = true
        bg.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -8).isActive = true
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}
