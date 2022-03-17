class Entity{
	constructor(x, y){
		this.xPos = x;
		this.yPos = y;
		this.width = 0;
		this.height = 0;
		this.entityImage = new Image();
	}

	setImage(imagePath){
		
		if(this.entityImage.src != null){
			delete this.entityImage.src;}
		this.entityImage.src = imagePath;
	}

	setSize(w, h) {
		this.width = w;
		this.height = h;
		this.customImageSize = true;
	}

	setPos(x, y) {
		this.xPos = x;
		this.yPos = y;
	}

	hasPos(x,y){
		return (this.xPos === x && this.yPos === y)
	}
}

class AnimatedEntity extends Entity{
	
	constructor(x, y) {
		super(x,y);
		this.newXPos = 0;
		this.newYPos = 0;
	}

	updatePosition(){
		this.xPos = this.newXPos;
		this.yPos = this.newYPos;
	}

	resetNewPosition(){
		this.newXPos = this.xPos;
		this.newYPos = this.yPos;
	}

	hasCollided(entity){
		return (this.xPos === entity.xPos && this.yPos === entity.yPos)

	}
}
class SandWall extends Entity{
	constructor(x,y){
		super(x,y);
		this.setImage("assets/sandwall.bmp");
	}
}
class SolidWall extends Entity{

	constructor(x,y){
		super(x,y);
		this.setImage("assets/solidwall.bmp");
	}
}

class BlastParticle extends AnimatedEntity{

	constructor(x,y){
		super(x,y);
		this.counter = 0;
		this.maxCounter = 10;
		this.finished = false;
		this.setImage("assets/blast_1.png");
	}

	update(){

		switch(this.counter)
		{
				case 2:
				this.setImage("assets/blast_2.png");
				break;
	
			case 4:
				this.setImage("assets/blast_3.png");
				break;

			case 8:
				this.setImage("assets/blast_4.png");
				break;

			case 10:
				this.setImage("assets/blast_5.png");
				this.finished = true;
				break;
		}


		this.counter++;
	}
}

class Gate extends Entity{
	constructor(x,y){
		super(x,y);
		this.setImage("assets/gate.png");
	}
}

class Key extends Entity{
	constructor(x,y){
		super(x,y);
		this.setImage("assets/key.gif");
	}
}
class Bomb extends Entity{
	constructor(x, y) {

		super(x,y);
		this.setImage("assets/bomb_1.png");

		this.exploded = false;
		this.countDown = 20;
	
	}

	update() {
		this.countDown--;

		if(this.countDown%2 === 0){
			this.setImage("assets/bomb_1.png");
		}
		else{
			this.setImage("assets/bomb_2.png");
		}

		if(this.countDown == 0)
		{
			this.exploded = true;
		}
	}
}

class Ozzy extends AnimatedEntity{

	constructor(x,y){
		super(x,y);
		this.setImage("assets/ozzy.gif");
		this.dropBomb = false;
	}

	update() {

		if (controller.gameMove == gameMoves.FIRE ) {
			this.dropBomb = true;
		}
		else if (controller.gameMove == gameMoves.RIGHT) {
			this.newXPos = this.xPos + 1;
		}
		else if (controller.gameMove == gameMoves.LEFT) {
			this.newXPos = this.xPos - 1;
		}
		else if (controller.gameMove == gameMoves.UP) {
			this.newYPos = this.yPos - 1;
		}
		else if (controller.gameMove == gameMoves.DOWN) {
			this.newYPos = this.yPos + 1;
		}

		controller.reset();
	}
}

class DeadOzzy extends AnimatedEntity{
	constructor(x, y){
		super(x,y);
		this.setImage("assets/bat.gif");
		this.finishedAnimation = false;
		this.countDown = 100;
	}

	update(){
		var iDirection = Math.floor(Math.random() * 5);

		if (iDirection === 1) {
			this.newXPos = this.xPos + 1;
		}
		else if (iDirection === 2) {
			this.newXPos = this.xPos - 1;
		}
		else if (iDirection === 3) {
			this.newYPos = this.yPos - 1;
		}
		else if (iDirection === 4) {
			this.newYPos = this.yPos + 1;
		}

		this.countDown--;

		if(this.countDown === 0){
			this.finishedAnimation = true;
		}
	}
}

class Beiber extends AnimatedEntity{
	constructor(x, y){
		super(x,y);
		this.setImage("assets/beibercalm.gif");
	
		this.ozzyPosX = 0;
		this.ozzyPosY = 0;
		this.angry = false;
		this.angerWaitCount = 0;
		this.angerWait = 4;

		this.calmBeiberTimer = 200;
		this.angryBeiberTimer = 100;
		this.beiberTemperCount = 0;
	}

	updateOzzyPos(x,y){
		this.ozzyPosX = x;
		this.ozzyPosY = y;
	}

	setAnger(anger){
		this.angry = anger
	}

	update(){
		// return ;

		this.beiberTemperCount++;
		if(this.angry && this.beiberTemperCount > this.angryBeiberTimer)
		{
			this.angry = false;
			this.beiberTemperCount = 0;
			this.setImage("assets/beibercalm.gif");
		}
		else if(!this.angry && this.beiberTemperCount > this.calmBeiberTimer)
		{
			this.angry = true;
			this.beiberTemperCount = 0;
			this.setImage("assets/beiberangry.gif");
		}

		if(this.angry){

			this.angerWaitCount++;

			if(this.angerWaitCount > this.angerWait){

				// do a directed drunkard walk
				if(Math.random() > 0.5){
					// move x
					var direction = Math.sign(this.ozzyPosX - this.xPos);
					this.newXPos = this.xPos + direction;
				}
				else{
					// move y
					var direction = Math.sign(this.ozzyPosY - this.yPos)
					this.newYPos = this.yPos + direction;
				}
				this.angerWaitCount = 0;
			}
		}
		else{
			var iDirection = Math.floor(Math.random() * 5);

			if (iDirection === 1) {
			    this.newXPos = this.xPos + 1;
			}
			else if (iDirection === 2) {
			    this.newXPos = this.xPos - 1;
			}
			else if (iDirection === 3) {
			    this.newYPos = this.yPos - 1;
			}
			else if (iDirection === 4) {
			    this.newYPos = this.yPos + 1;
			}
		}
	}
}
