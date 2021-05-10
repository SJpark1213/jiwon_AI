//
//  RainbowCircle.swift
//  CustomShapeforSlider
//
//  Created by 강지원 on 2020/10/28.
//

import UIKit

class RainbowCircle: UIView {
    private var radius: CGFloat {
        return frame.width>frame.height ? frame.height/2 : frame.width/2
    }

    private var stroke: CGFloat = 10
    private var padding: CGFloat = 5

    //MARK: - Drawing
    override func draw(_ rect: CGRect) {
        super.draw(rect)
        drawRainbowCircle(outerRadius: radius - padding, innerRadius: radius - stroke - padding, resolution: 1)
    }

    init(frame: CGRect, lineHeight: CGFloat) {
        super.init(frame: frame)
        stroke = lineHeight
    }

    required init?(coder aDecoder: NSCoder) { super.init(coder: aDecoder) }

    /*
     Resolution should be between 0.1 and 1
     */
    private func drawRainbowCircle(outerRadius: CGFloat, innerRadius: CGFloat, resolution: Float) {
        guard let context = UIGraphicsGetCurrentContext() else { return }
        context.saveGState()
        context.translateBy(x: self.bounds.midX, y: self.bounds.midY) //Move context to center

        let subdivisions:CGFloat = CGFloat(resolution * 512) //Max subdivisions of 512

        let innerHeight = (CGFloat.pi*innerRadius)/subdivisions //height of the inner wall for each segment
        let outterHeight = (CGFloat.pi*outerRadius)/subdivisions

        let segment = UIBezierPath()
        segment.move(to: CGPoint(x: innerRadius, y: -innerHeight/2))
        segment.addLine(to: CGPoint(x: innerRadius, y: innerHeight/2))
        segment.addLine(to: CGPoint(x: outerRadius, y: outterHeight/2))
        segment.addLine(to: CGPoint(x: outerRadius, y: -outterHeight/2))
        segment.close()

        //Draw each segment and rotate around the center
        for i in 0 ..< Int(ceil(subdivisions)) {
            UIColor(hue: CGFloat(i)/subdivisions, saturation: 1, brightness: 1, alpha: 1).set()
            segment.fill()
            //let lineTailSpace = CGFloat.pi*2*outerRadius/subdivisions  //The amount of space between the tails of each segment
            let lineTailSpace = CGFloat.pi*2*outerRadius/subdivisions
            segment.lineWidth = lineTailSpace //allows for seemless scaling
            segment.stroke()

            //Rotate to correct location
            let rotate = CGAffineTransform(rotationAngle: -(CGFloat.pi*2/subdivisions)) //rotates each segment
            segment.apply(rotate)
        }

        context.translateBy(x: -self.bounds.midX, y: -self.bounds.midY) //Move context back to original position
        context.restoreGState()
    }

}
