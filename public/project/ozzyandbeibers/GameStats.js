class GameStats{
        constructor(ctx, width,height){
            this.context = ctx;
            this.context.font = "40px Arial";
            this.level = 0;
            this.score = 0;
            this.ozzies = 3;
            this.isUpdated = true;
            this.width = width;
            this.height = height;
        }

        updateScore(score){
            if(this.score != score){
                this.score = score;
                this.isUpdated = true;
            }
        }

        updateLevel(level){
            this.level = level;
            this.isUpdated = true;
        }

        updateOzzies(ozzies){
            this.ozzies = ozzies;
            this.isUpdated = true;
        }

        draw(){
            if(this.isUpdated){
                this.context.clearRect(0, 0, this.width, this.height);
                this.context.rect(0, 0, this.pageSize+1, this.pageSize+1);
                this.context.stroke();


                this.context.font = "30px Verdana";
                this.context.fillStyle = 'Yellow';
                this.context.fillText("Level: "+this.level, 30, 40);
                this.context.fillText("Score: "+this.score, 30, 90);
                this.context.fillText("Ozzies: "+this.ozzies, 30, 140);
            }
        }
}