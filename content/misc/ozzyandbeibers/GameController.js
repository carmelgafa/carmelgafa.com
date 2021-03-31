const gameMoves = {
	NONE: "none",
	UP: "up",
	DOWN: "down",
    LEFT: "left",
    RIGHT: "right",
    FIRE: "fire"
}

class GameController{
    
    constructor(){
        this.gameMove = gameMoves.NONE;
    }

    reset(){
        this.gameMove = gameMoves.NONE;
    }

    onKeydown(event) {
        var keycode = event.code

        switch (keycode) {
            case 'Space':       this.gameMove = gameMoves.FIRE; break;
            case 'ArrowLeft':   this.gameMove = gameMoves.LEFT; break;
            case 'ArrowUp':     this.gameMove = gameMoves.UP; break;
            case 'ArrowRight':  this.gameMove = gameMoves.RIGHT; break;
            case 'ArrowDown':   this.gameMove = gameMoves.DOWN; break;
        }
    }
}