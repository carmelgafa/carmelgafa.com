const gameStates = {
	LOST: "lost",
	OZZYHIT: "ozzyhit",
	WON: "won",
	PLAYING: "playing",
}

class Game{
	constructor(ctx){
		this.gameState = gameStates.PLAYING;
		this.context = ctx;
		this.hardWalls = new Array();
		this.sandWalls = new Array();
		this.bombs = new Array();
		this.bombBlasts = new Array();
		this.blastParticles=new Array();
		this.gate = null;
		this.beibers = new Array();
		this.ozzy = null;
		this.lives = 3;
		this.score = 0;
		this.blocksNumber = 11;
		this.pageSize = 0;
		this.blockSize = 0;
		
		this.initializeHardBlocks();
	}

	clearGame(){
		this.sandWalls.length = 0;
		this.bombs.length = 0;
		this.bombBlasts.length = 0;
		this.gate = null;
		this.beibers.length = 0;
		this.blastParticles.length = 0;
		this.ozzy.null;
	}

	buildLevel(definition){
		var thisObject = this;

		definition.sand.forEach(addSandBlock);
		function addSandBlock(pos) {
			var block = new SandWall(pos[0], pos[1]);
			thisObject.sandWalls.push(block);
		}


		definition.beibers.forEach(addBeiber);
		function addBeiber(pos) {
			var block = new Beiber(pos[0], pos[1]);
			thisObject.beibers.push(block);
		}

		this.gate = new Gate(definition.gate[0], definition.gate[1])

		this.ozzy = new Ozzy(definition.ozzy[0], definition.ozzy[1]);

		this.gameState = gameStates.PLAYING;
	}

	initializeHardBlocks(){
		for (var i = 0; i < this.blocksNumber; i += 2) {
			for (var j = 0; j < this.blocksNumber; j += 2) {
				var block = new SolidWall(i, j);
				this.hardWalls.push(block);
			}
		}
	}

	canMovePos(x, y) {
		if (
			 x < 0 || 
			 x > this.pageSize / this.blockSize - 1 ||
			 y < 0 || 
			 y > this.pageSize / this.blockSize - 1)
			return false;
	
		for (i = 0; i < this.hardWalls.length; i++) {
			if (x === this.hardWalls[i].xPos && y === this.hardWalls[i].yPos)
				return false;
		}
	
		for (var i = 0; i < this.sandWalls.length; i++) {
			if (x === this.sandWalls[i].xPos && y === this.sandWalls[i].yPos)
				return false;
		}
		return true;
	}

	drawEntity(entity) {
		this.context.drawImage(entity.entityImage, 
			(entity.xPos * this.blockSize)+1, 
			(entity.yPos * this.blockSize)+1, 
			this.blockSize, 
			this.blockSize);
	}

	moveEntity(entity) {

		entity.update();
		if (this.canMovePos(entity.newXPos, entity.newYPos) === true) {
			entity.updatePosition();
		}
		else {
			entity.resetNewPosition();
		}
	}

	updateBombs(){
		for (var i = 0; i < this.bombs.length; i++) {
			this.bombs[i].update();
	
			if (this.bombs[i].exploded === true) {
				var x = this.bombs[i].xPos;
				var y = this.bombs[i].yPos;

				this.blastBomb(x,y)

				// remove the bomb as it is now exploded
				this.bombs.splice(i, 1);
			}
		}
	}

	canCreateBlast(x,y){
		for (var i = 0; i < this.hardWalls.length; i++) {
			if(this.hardWalls[i].hasPos(x, y)){
				return false;
			}
		}

		for (i = 0; i < this.sandWalls.length; i++) {
			if(this.sandWalls[i].hasPos(x,y)){
				this.sandWalls.splice(i, 1);
				this.score += 20;
				return false;
			}
		}

		for (i = 0; i < this.bombs.length; i++) {
			if(this.bombs[i].hasPos(x,y)){
				this.bombs[i].exploded = true;
				this.score += 5;
				return false;
			}
		}

		for (i = 0; i < this.beibers.length; i++) {
			if(this.beibers[i].hasPos(x,y)){
				this.score += 100;
				this.beibers.splice(i, 1);
				return false;
			}
		}
		return true;
	}

	blastBomb(x,y){
		this.blastParticles.push(new BlastParticle(x,y));

		for(var i=1; i<3; i++){
			if(this.canCreateBlast(x+i, y)){
				this.blastParticles.push(new BlastParticle(x+i, y));
			}
			else{
				this.blastParticles.push(new BlastParticle(x+i, y));
				break;
			}
		}

		for(var i=1; i<3; i++){
			if(this.canCreateBlast(x-i, y)){
				this.blastParticles.push(new BlastParticle(x-i, y));
			}
			else{
				this.blastParticles.push(new BlastParticle(x-i, y));
				break;
			}
		}

		for(var i=1; i<3; i++){
			if(this.canCreateBlast(x, y+i)){
				this.blastParticles.push(new BlastParticle(x, y+i));
			}
			else{
				this.blastParticles.push(new BlastParticle(x, y+i));
				break;
			}
		}

		for(var i=1; i<3; i++){
			if(this.canCreateBlast(x, y-i)){
				this.blastParticles.push(new BlastParticle(x, y-i));
			}
			else{
				this.blastParticles.push(new BlastParticle(x, y-i));
				break;
			}
		}
	}


	updateBlasts(){

		let remainingBlasts = new Array();
		for(var i = 0; i < this.blastParticles.length; i++){
			
			this.blastParticles[i].update();
			

			if(this.blastParticles[i].hasCollided(this.ozzy)){
				this.gameState = gameStates.OZZYHIT;
				var x = this.ozzy.xPos;
				var y = this.ozzy.yPos;
				this.ozzy = new DeadOzzy(x,y)
			}

			if(!this.blastParticles[i].finished){
				remainingBlasts.push(this.blastParticles[i]);
			}

			let remainingBeibers = new Array();
			for(let i = 0; i < this.beibers.length; i++){
				if(!this.blastParticles[i].hasCollided(this.beibers[i])){
					remainingBeibers.push(this.beibers[i]);
				}
				else{
					this.score += 100;
				}
			}
			this.beibers = remainingBeibers;
		
		}
		this.blastParticles = remainingBlasts;

		return false;
	}

	update(){

		this.updateBlasts();

		if(this.gameState == gameStates.OZZYHIT){
			this.ozzy.update();
			this.moveEntity(this.ozzy);
			if(this.ozzy.finishedAnimation){
				this.gameState = gameStates.LOST;
			}
		}
		else{
			if(this.gameState === gameStates.PLAYING){
				if (this.ozzy.dropBomb === true) {
					var newBomb= new Bomb(this.ozzy.xPos, this.ozzy.yPos);
					this.bombs.push(newBomb);
					this.ozzy.dropBomb = false;
				}
				else{
					this.moveEntity(this.ozzy);
				}

				for (var j = 0; j < this.beibers.length; j++) {
					//check beibers with bombs
					this.beibers[j].updateOzzyPos(this.ozzy.xPos, this.ozzy.yPos)
					this.moveEntity(this.beibers[j]);

					// and with ozz
					if(this.beibers[j].hasCollided(this.ozzy)){
						this.gameState = gameStates.OZZYHIT;
						var x = this.ozzy.xPos;
						var y = this.ozzy.yPos;
						this.ozzy = new DeadOzzy(x,y)
					}
				}

				if(this.ozzy.hasCollided(this.gate)){
					this.score += 150;
					this.gameState = gameStates.WON;
				}
				this.updateBombs();
			}
		}
	}

	draw(){
		this.context.clearRect(0, 0, this.pageSize, this.pageSize);
		this.context.rect(0, 0, this.pageSize+1, this.pageSize+1);
		this.context.stroke();

		this.drawEntity(this.gate);

		for (var i = 0; i < this.blastParticles.length; i++) {
			this.drawEntity(this.blastParticles[i]);
		}

		for (var i = 0; i < this.hardWalls.length; i++) {
			this.drawEntity(this.hardWalls[i]);
		}

		for (var i = 0; i < this.sandWalls.length; i++) {
			this.drawEntity(this.sandWalls[i]);
		}

		for (var i = 0; i < this.beibers.length; i++) {
			this.drawEntity(this.beibers[i]);
		}

		for (i = 0; i < this.bombs.length; i++) {
			this.drawEntity(this.bombs[i]);
		}
		// Draw Ozzy
		this.drawEntity(this.ozzy);
	}
}