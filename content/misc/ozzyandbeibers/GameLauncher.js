const mainCanvas = document.getElementById('canvas1');
const livesCanvas = document.getElementById('livescanvas');
const scoreCanvas = document.getElementById('scorecanvas');
const mainCtx = mainCanvas.getContext('2d');
const scoreCtx = scoreCanvas.getContext('2d');

let score = 0;

class GameLauncher {
	constructor() {
		this.game = new Game(mainCtx);
		this.gameStats = new GameStats(scoreCtx, scoreCanvas.width, scoreCanvas.height);
		this.current_level = 1;
		this.gameStats.updateLevel(this.current_level);
	}

	startGame() {
		this.executeLevel();
	}

	executeLevel() {
		var level = 'level_' + this.current_level;
		this.game.buildLevel(levels[level]);
		this.run();
	}

	run() {
		var deltaTimeMS = 100;

		var thisObject = this;

		this.intervalId = self.setInterval(function () {
			thisObject.update();
			thisObject.draw();
		}, deltaTimeMS);
	}

	processLoss() {
		this.game.lives--;
		this.gameStats.updateOzzies(this.game.lives);
		if(this.game.lives === 0){
			window.location = 'lost.html'
		}
		this.game.clearGame();
		this.executeLevel()
	}

	processWin() {
		console.log('yahoooo');
		this.current_level++;
		this.gameStats.updateLevel(this.current_level);
		this.game.clearGame();
		this.executeLevel()
	}

	update() {
		switch (this.game.gameState) {
			case gameStates.LOST:
				clearInterval(this.intervalId);
				this.processLoss();
				break;

			case gameStates.WON:
				clearInterval(this.intervalId);
				this.processWin();
				break;

			case gameStates.PLAYING:
			case gameStates.OZZYHIT:
				this.gameStats.updateScore(this.game.score);
				this.game.update();
				break;
		}
	}

	draw() {
		this.game.draw();
		this.gameStats.draw();
	}

	calcPageSize() {
		this.game.calcPageSize(mainCtx);
	}
}

const g = new GameLauncher();

var controller = new GameController();
window.addEventListener('keydown', function (event) {
	controller.onKeydown(event);
}, false);

window.addEventListener('load', loadHandler);
window.addEventListener("resize", calculateSize);

function calculateSize() {
	var dwidth = (window.innerWidth * 0.4) - 40;

	g.game.blockSize = Math.floor(dwidth / g.game.blocksNumber);

	g.game.pageSize = (g.game.blockSize * g.game.blocksNumber);

	mainCanvas.width = g.game.pageSize;
	mainCanvas.height = g.game.pageSize;

	dwidth = (window.innerWidth * 0.30);
	livesCanvas.width = dwidth;
	scoreCanvas.width = dwidth;
};


function loadHandler() {
	calculateSize();
	g.startGame();
};